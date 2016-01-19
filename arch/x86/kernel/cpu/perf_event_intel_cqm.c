/*
 * Intel Cache Quality-of-Service Monitoring (CQM) support.
 *
 * Based very, very heavily on work by Peter Zijlstra.
 */

#include <linux/perf_event.h>
#include <linux/slab.h>
#include <asm/cpu_device_id.h>
#include "perf_event.h"

#define MSR_IA32_PQR_ASSOC	0x0c8f
#define MSR_IA32_QM_CTR		0x0c8e
#define MSR_IA32_QM_EVTSEL	0x0c8d

/*
 * MBM Counter is 24bits wide. MBM_CNTR_MAX defines max counter
 * value
 */
#define MBM_CNTR_MAX		0xffffff

/*
 *  Maximum number of MBM event types supported
 */
#define MAX_MBM_EVENT_TYPES 1
/*
 * Expected time interval in ms between consecutive MSR reads for a given rmid
 */
#define MBM_TIME_DELTA_EXP	1000

/*
 *  Minimum time interval in ms between consecutive MSR reads for a given rmid
 */
#define MBM_TIME_DELTA_MIN	(100 * MAX_MBM_EVENT_TYPES)

/*
 * Minimum size for sliding window i.e. the minimum monitoring period for
 * application(s). This fifo_size can be used for short duration monitoring
 * since short duration monitoring will have less number of samples.
 * Corresponding sliding window duration will be 10sec. mbm_window_size
 * variable is used to set the current monitoring duration.
 */
#define MBM_FIFO_SIZE_MIN	10
/*
 * Maximum size for sliding window i.e. the maximum monitoring period that is
 * supported. Corresponsing sliding window  duration for this fifo_size is
 * 300sec. Typically long duration monitoring session can use this window size.
 */
#define MBM_FIFO_SIZE_MAX	300
/*
 * mbm_window_size is used to set current monitoring period. This means
 * mbm_window_size defines the number of profiled samples to be stored in
 * sliding window i.e. mbm_fifo.
 */
static u32 mbm_window_size = MBM_FIFO_SIZE_MIN;
static u32 cqm_max_rmid = -1;
static unsigned int cqm_l3_scale; /* supposedly cacheline size */
static bool cqm_llc_occ, is_mbm;
static u16  mbm_socket_max;

/**
 * struct intel_pqr_state - State cache for the PQR MSR
 * @rmid:		The cached Resource Monitoring ID
 * @closid:		The cached Class Of Service ID
 * @rmid_usecnt:	The usage counter for rmid
 *
 * The upper 32 bits of MSR_IA32_PQR_ASSOC contain closid and the
 * lower 10 bits rmid. The update to MSR_IA32_PQR_ASSOC always
 * contains both parts, so we need to cache them.
 *
 * The cache also helps to avoid pointless updates if the value does
 * not change.
 */
struct intel_pqr_state {
	u32			rmid;
	u32			closid;
	int			rmid_usecnt;
};

/*
 * The cached intel_pqr_state is strictly per CPU and can never be
 * updated from a remote CPU. Both functions which modify the state
 * (intel_cqm_event_start and intel_cqm_event_stop) are called with
 * interrupts disabled, which is sufficient for the protection.
 */
static DEFINE_PER_CPU(struct intel_pqr_state, pqr_state);
static DEFINE_PER_CPU(struct mbm_pmu *, mbm_pmu);

/**
 * struct mbm_pmu - mbm events per cpu
 * @n_active:       number of active events for this pmu
 * @active_list:    linked list for perf events for this pmu
 * @pmu:            pmu per cpu
 * @timer_interval: pmu's hrtimer period
 * @hrtimer:        periodic high resolution timer for this pmu
 *                  intel_mbm_event_update is the callback function that gets
 *                  triggered by hrtimer and profiles for a new mbm sample.
 */
struct mbm_pmu {
	int              n_active;
	struct list_head active_list;
	struct pmu       *pmu;
	ktime_t          timer_interval;
	struct hrtimer   hrtimer;
};

/**
 * struct sample - mbm event's (local or total) data
 * @bytes:         previous MSR value
 * @runavg:        running average of memory bandwidth
 * @prev_time:     time stamp of previous sample i.e. {bytes, runavg}
 * @index:         current sample number
 * @fifoin:        sliding window counter to store the sample
 * @fifoout:       start of the sliding window to calculate  bandwidh sum
 */
struct sample {
	u64 bytes;
	u64 runavg;
	ktime_t prev_time;
	u32 index;
	u32 mbmfifo[MBM_FIFO_SIZE_MAX];
	u32  fifoin;
	u32  fifoout;
};

/*
 * samples profiled for total memory bandwidth type events
 */
static struct sample *mbm_total;

/*
 * samples profiled for local memory bandwidth type events
 */
static struct sample *mbm_local;

#define pkg_id	topology_physical_package_id(smp_processor_id())
/*
 * rmid_2_index returns the index for the rmid in mbm_local/mbm_total array.
 * mbm_total[] and mbm_local[] are linearly indexed by core# * max number of
 * rmids per socket, an example is given below
 * RMID1 of Socket0:  vrmid  = 1
 * RMID1 of Socket1:  vrmid =  1 * cqm_max_rmid + 1
 * RMID1 of Socket2:  vrmid =  2 * cqm_max_rmid + 1
 */
#define rmid_2_index(rmid)  (pkg_id * cqm_max_rmid + rmid)

static enum hrtimer_restart mbm_hrtimer_handle(struct hrtimer *hrtimer);

/*
 * Protects cache_cgroups and cqm_rmid_free_lru and cqm_rmid_limbo_lru.
 * Also protects event->hw.cqm_rmid
 *
 * Hold either for stability, both for modification of ->hw.cqm_rmid.
 */
static DEFINE_MUTEX(cache_mutex);
static DEFINE_RAW_SPINLOCK(cache_lock);

/*
 * Groups of events that have the same target(s), one RMID per group.
 */
static LIST_HEAD(cache_groups);

/*
 * Mask of CPUs for reading CQM values. We only need one per-socket.
 */
static cpumask_t cqm_cpumask;

#define RMID_VAL_ERROR		(1ULL << 63)
#define RMID_VAL_UNAVAIL	(1ULL << 62)

#define QOS_L3_OCCUP_EVENT_ID	0x01
/*
 * MBM Event IDs as defined in SDM section 17.15.5
 * Event IDs are used to program EVTSEL MSRs before reading mbm event counters
 */
enum mbm_evt_type {
	QOS_MBM_TOTAL_EVENT_ID = 0x02,
	QOS_MBM_LOCAL_EVENT_ID,
	QOS_MBM_TOTAL_AVG_EVENT_ID,
	QOS_MBM_LOCAL_AVG_EVENT_ID,
};

#define QOS_MBM_AVG_EVENT_MASK 0x04
#define QOS_MBM_LOCAL_EVENT_MASK 0x01

/*
 * This is central to the rotation algorithm in __intel_cqm_rmid_rotate().
 *
 * This rmid is always free and is guaranteed to have an associated
 * near-zero occupancy value, i.e. no cachelines are tagged with this
 * RMID, once __intel_cqm_rmid_rotate() returns.
 */
static u32 intel_cqm_rotation_rmid;

#define INVALID_RMID		(-1)

/*
 * Is @rmid valid for programming the hardware?
 *
 * rmid 0 is reserved by the hardware for all non-monitored tasks, which
 * means that we should never come across an rmid with that value.
 * Likewise, an rmid value of -1 is used to indicate "no rmid currently
 * assigned" and is used as part of the rotation code.
 */
static inline bool __rmid_valid(u32 rmid)
{
	if (!rmid || rmid == INVALID_RMID)
		return false;

	return true;
}

static u64 __rmid_read(u32 rmid)
{
	u64 val;

	/*
	 * Ignore the SDM, this thing is _NOTHING_ like a regular perfcnt,
	 * it just says that to increase confusion.
	 */
	wrmsr(MSR_IA32_QM_EVTSEL, QOS_L3_OCCUP_EVENT_ID, rmid);
	rdmsrl(MSR_IA32_QM_CTR, val);

	/*
	 * Aside from the ERROR and UNAVAIL bits, assume this thing returns
	 * the number of cachelines tagged with @rmid.
	 */
	return val;
}

enum rmid_recycle_state {
	RMID_YOUNG = 0,
	RMID_AVAILABLE,
	RMID_DIRTY,
};

struct cqm_rmid_entry {
	u32 rmid;
	enum rmid_recycle_state state;
	struct list_head list;
	unsigned long queue_time;
	bool is_cqm;
	bool is_multi_event;
};

static void intel_cqm_free_rmid(struct cqm_rmid_entry *entry);

/*
 * cqm_rmid_free_lru - A least recently used list of RMIDs.
 *
 * Oldest entry at the head, newest (most recently used) entry at the
 * tail. This list is never traversed, it's only used to keep track of
 * the lru order. That is, we only pick entries of the head or insert
 * them on the tail.
 *
 * All entries on the list are 'free', and their RMIDs are not currently
 * in use. To mark an RMID as in use, remove its entry from the lru
 * list.
 *
 *
 * cqm_rmid_limbo_lru - list of currently unused but (potentially) dirty RMIDs.
 *
 * This list is contains RMIDs that no one is currently using but that
 * may have a non-zero occupancy value associated with them. The
 * rotation worker moves RMIDs from the limbo list to the free list once
 * the occupancy value drops below __intel_cqm_threshold.
 *
 * Both lists are protected by cache_mutex.
 */
static LIST_HEAD(cqm_rmid_free_lru);
static LIST_HEAD(cqm_rmid_limbo_lru);

/*
 * We use a simple array of pointers so that we can lookup a struct
 * cqm_rmid_entry in O(1). This alleviates the callers of __get_rmid()
 * and __put_rmid() from having to worry about dealing with struct
 * cqm_rmid_entry - they just deal with rmids, i.e. integers.
 *
 * Once this array is initialized it is read-only. No locks are required
 * to access it.
 *
 * All entries for all RMIDs can be looked up in the this array at all
 * times.
 */
static struct cqm_rmid_entry **cqm_rmid_ptrs;

static inline struct cqm_rmid_entry *__rmid_entry(u32 rmid)
{
	struct cqm_rmid_entry *entry;

	entry = cqm_rmid_ptrs[rmid];
	WARN_ON(entry->rmid != rmid);

	return entry;
}

/**
 * mbm_reset_stats - reset stats for a given rmid for the current cpu
 * @rmid:	rmid value
 *
 * vrmid: array index for mbm_total or mbm_local of the current core for the
 * given rmid
 *
 * mbs_total[] and mbm_local[] are linearly indexed by core number * max number
 * rmids per socket, an example is given below
 * RMID1 of Socket0:  vrmid  = 1
 * RMID1 of Socket1:  vrmid =  1 * CQM_MAX_RMID + 1
 * RMID1 of Socket2:  vrmid =  2 * CQM_MAX_RMID + 1
 */
static void mbm_reset_stats(u32 rmid)
{
	u32  i, vrmid;

	if (!is_mbm)
		return;
	for (i=0; i < mbm_socket_max; i++) {
		vrmid =  i * cqm_max_rmid + rmid;
		memset(&mbm_local[vrmid], 0, sizeof(struct sample));
		memset(&mbm_total[vrmid], 0, sizeof(struct sample));
	}
}

/*
 * Returns < 0 on fail.
 *
 * We expect to be called with cache_mutex held.
 */
static u32 __get_rmid(void)
{
	struct cqm_rmid_entry *entry;

	lockdep_assert_held(&cache_mutex);

	if (list_empty(&cqm_rmid_free_lru))
		return INVALID_RMID;

	entry = list_first_entry(&cqm_rmid_free_lru, struct cqm_rmid_entry, list);
	list_del(&entry->list);

	return entry->rmid;
}

static void __put_rmid(u32 rmid)
{
	struct cqm_rmid_entry *entry;

	lockdep_assert_held(&cache_mutex);

	WARN_ON(!__rmid_valid(rmid));
	entry = __rmid_entry(rmid);

	entry->queue_time = jiffies;
	entry->state = RMID_YOUNG;
	mbm_reset_stats(rmid);

	/*
	 * If the RMID is used for measuring LLC_OCCUPANCY, put it in
	 * cqm_rmid_limbo_lru so that it gets recycled. Otherwise, RMID
	 * is put in free list and is immediately available for reuse
	 */
	if (entry->is_cqm)
		list_add_tail(&entry->list, &cqm_rmid_limbo_lru);
	else
		intel_cqm_free_rmid(entry);

}

static int intel_cqm_setup_rmid_cache(void)
{
	struct cqm_rmid_entry *entry;
	unsigned int nr_rmids;
	int r = 0;

	nr_rmids = cqm_max_rmid + 1;
	cqm_rmid_ptrs = kmalloc(sizeof(struct cqm_rmid_entry *) *
				nr_rmids, GFP_KERNEL);
	if (!cqm_rmid_ptrs)
		return -ENOMEM;

	for (; r <= cqm_max_rmid; r++) {
		struct cqm_rmid_entry *entry;

		entry = kmalloc(sizeof(*entry), GFP_KERNEL);
		if (!entry)
			goto fail;

		INIT_LIST_HEAD(&entry->list);
		entry->rmid = r;
		entry->is_cqm = false;
		cqm_rmid_ptrs[r] = entry;

		list_add_tail(&entry->list, &cqm_rmid_free_lru);
	}

	/*
	 * RMID 0 is special and is always allocated. It's used for all
	 * tasks that are not monitored.
	 */
	entry = __rmid_entry(0);
	list_del(&entry->list);

	mutex_lock(&cache_mutex);
	intel_cqm_rotation_rmid = __get_rmid();
	mutex_unlock(&cache_mutex);

	return 0;
fail:
	while (r--)
		kfree(cqm_rmid_ptrs[r]);

	kfree(cqm_rmid_ptrs);
	return -ENOMEM;
}

/*
 * Determine if @a and @b measure the same set of tasks.
 *
 * If @a and @b measure the same set of tasks then we want to share a
 * single RMID.
 */
static bool __match_event(struct perf_event *a, struct perf_event *b)
{
	/* Per-cpu and task events don't mix */
	if ((a->attach_state & PERF_ATTACH_TASK) !=
	    (b->attach_state & PERF_ATTACH_TASK))
		return false;

#ifdef CONFIG_CGROUP_PERF
	if (a->cgrp != b->cgrp)
		return false;
#endif

	/* If not task event, we're machine wide */
	if (!(b->attach_state & PERF_ATTACH_TASK))
		return true;

	/*
	 * Events that target same task are placed into the same cache group.
	 */
	if (a->hw.target == b->hw.target) {
		if (a->attr.config  != b->attr.config) {
			struct cqm_rmid_entry *entry;

				entry = __rmid_entry(a->hw.cqm_rmid);
				entry->is_multi_event = true;
		}
		return true;
	}

	/*
	 * Are we an inherited event?
	 */
	if (b->parent == a)
		return true;

	return false;
}

#ifdef CONFIG_CGROUP_PERF
static inline struct perf_cgroup *event_to_cgroup(struct perf_event *event)
{
	if (event->attach_state & PERF_ATTACH_TASK)
		return perf_cgroup_from_task(event->hw.target, event->ctx);

	return event->cgrp;
}
#endif

/*
 * Determine if @a's tasks intersect with @b's tasks
 *
 * There are combinations of events that we explicitly prohibit,
 *
 *		   PROHIBITS
 *     system-wide    -> 	cgroup and task
 *     cgroup 	      ->	system-wide
 *     		      ->	task in cgroup
 *     task 	      -> 	system-wide
 *     		      ->	task in cgroup
 *
 * Call this function before allocating an RMID.
 */
static bool __conflict_event(struct perf_event *a, struct perf_event *b)
{
#ifdef CONFIG_CGROUP_PERF
	/*
	 * We can have any number of cgroups but only one system-wide
	 * event at a time.
	 */
	if (a->cgrp && b->cgrp) {
		struct perf_cgroup *ac = a->cgrp;
		struct perf_cgroup *bc = b->cgrp;

		/*
		 * This condition should have been caught in
		 * __match_event() and we should be sharing an RMID.
		 */
		WARN_ON_ONCE(ac == bc);

		if (cgroup_is_descendant(ac->css.cgroup, bc->css.cgroup) ||
		    cgroup_is_descendant(bc->css.cgroup, ac->css.cgroup))
			return true;

		return false;
	}

	if (a->cgrp || b->cgrp) {
		struct perf_cgroup *ac, *bc;

		/*
		 * cgroup and system-wide events are mutually exclusive
		 */
		if ((a->cgrp && !(b->attach_state & PERF_ATTACH_TASK)) ||
		    (b->cgrp && !(a->attach_state & PERF_ATTACH_TASK)))
			return true;

		/*
		 * Ensure neither event is part of the other's cgroup
		 */
		ac = event_to_cgroup(a);
		bc = event_to_cgroup(b);
		if (ac == bc)
			return true;

		/*
		 * Must have cgroup and non-intersecting task events.
		 */
		if (!ac || !bc)
			return false;

		/*
		 * We have cgroup and task events, and the task belongs
		 * to a cgroup. Check for for overlap.
		 */
		if (cgroup_is_descendant(ac->css.cgroup, bc->css.cgroup) ||
		    cgroup_is_descendant(bc->css.cgroup, ac->css.cgroup))
			return true;

		return false;
	}
#endif
	/*
	 * If one of them is not a task, same story as above with cgroups.
	 */
	if (!(a->attach_state & PERF_ATTACH_TASK) ||
	    !(b->attach_state & PERF_ATTACH_TASK))
		return true;

	/*
	 * Must be non-overlapping.
	 */
	return false;
}

struct rmid_read {
	u32 rmid;
	atomic64_t value;
	enum mbm_evt_type evt_type;
};

static void __intel_cqm_event_count(void *info);

/*
 * Exchange the RMID of a group of events.
 */
static u32 intel_cqm_xchg_rmid(struct perf_event *group, u32 rmid)
{
	struct perf_event *event;
	struct list_head *head = &group->hw.cqm_group_entry;
	u32 old_rmid = group->hw.cqm_rmid;

	lockdep_assert_held(&cache_mutex);

	/*
	 * If our RMID is being deallocated, perform a read now.
	 */
	if (__rmid_valid(old_rmid) && !__rmid_valid(rmid)) {
		struct rmid_read rr = {
			.value = ATOMIC64_INIT(0),
			.rmid = old_rmid,
		};

		on_each_cpu_mask(&cqm_cpumask, __intel_cqm_event_count,
				 &rr, 1);
		local64_set(&group->count, atomic64_read(&rr.value));
	}

	raw_spin_lock_irq(&cache_lock);

	group->hw.cqm_rmid = rmid;
	list_for_each_entry(event, head, hw.cqm_group_entry)
		event->hw.cqm_rmid = rmid;

	raw_spin_unlock_irq(&cache_lock);

	return old_rmid;
}

/*
 * If we fail to assign a new RMID for intel_cqm_rotation_rmid because
 * cachelines are still tagged with RMIDs in limbo, we progressively
 * increment the threshold until we find an RMID in limbo with <=
 * __intel_cqm_threshold lines tagged. This is designed to mitigate the
 * problem where cachelines tagged with an RMID are not steadily being
 * evicted.
 *
 * On successful rotations we decrease the threshold back towards zero.
 *
 * __intel_cqm_max_threshold provides an upper bound on the threshold,
 * and is measured in bytes because it's exposed to userland.
 */
static unsigned int __intel_cqm_threshold;
static unsigned int __intel_cqm_max_threshold;

/*
 * Test whether an RMID has a zero occupancy value on this cpu.
 */
static void intel_cqm_stable(void *arg)
{
	struct cqm_rmid_entry *entry;

	list_for_each_entry(entry, &cqm_rmid_limbo_lru, list) {
		if (entry->state != RMID_AVAILABLE)
			break;

		if (__rmid_read(entry->rmid) > __intel_cqm_threshold)
			entry->state = RMID_DIRTY;
	}
}

/*
 * If we have group events waiting for an RMID that don't conflict with
 * events already running, assign @rmid.
 */
static bool intel_cqm_sched_in_event(u32 rmid)
{
	struct perf_event *leader, *event;

	lockdep_assert_held(&cache_mutex);

	leader = list_first_entry(&cache_groups, struct perf_event,
				  hw.cqm_groups_entry);
	event = leader;

	list_for_each_entry_continue(event, &cache_groups,
				     hw.cqm_groups_entry) {
		if (__rmid_valid(event->hw.cqm_rmid))
			continue;

		if (__conflict_event(event, leader))
			continue;

		intel_cqm_xchg_rmid(event, rmid);
		return true;
	}

	return false;
}

static void intel_cqm_free_rmid(struct cqm_rmid_entry *entry)
{
	/*
	 * The rotation RMID gets priority if it's currently invalid.
	 *
	 * In which case, skip adding the RMID to the the free lru.
	 */
	 if (!__rmid_valid(intel_cqm_rotation_rmid)) {
		intel_cqm_rotation_rmid = entry->rmid;
		return;
	}

	/*
	 * If we have groups waiting for RMIDs, hand them one now
	 * provided they don't conflict.
	 */
	if (intel_cqm_sched_in_event(entry->rmid))
		return;

	/*
	 * Otherwise place it onto the free list.
	 */
	list_add_tail(&entry->list, &cqm_rmid_free_lru);
}

/*
 * Slide the window by 1 and calculate the sum of the last
 * mbm_window_size-1  bandwidth  values.
 * fifoout is the current position of the window.
 * Increment the fifoout by 1 to slide the window by 1.
 *
 * Calcalute the bandwidth from ++fifiout  to ( ++fifoout + mbm_window_size -1)
 * e.g.fifoout =1;   Bandwidth1 Bandwidth2 ..... Bandwidthn are the
 * sliding window values where n is size of the sliding window
 * bandwidth sum:  val  =  Bandwidth2 + Bandwidth3 + .. Bandwidthn
 */

static u32 __mbm_fifo_sum_lastn_out(struct sample *bw_stat)
{
	u32 val = 0, i, j, index;

	if (++bw_stat->fifoout >=  mbm_window_size)
		bw_stat->fifoout =  0;
	index =  bw_stat->fifoout;
	for (i = 0; i < mbm_window_size - 1; i++) {
		if ((index + i) >= mbm_window_size)
			j = index + i - mbm_window_size;
		else
			j = index + i;
		val += bw_stat->mbmfifo[j];
	}
	return val;
}

/*
 * store current sample's bw value in sliding window at the
 * location fifoin. Increment fifoin. Check if fifoin has reached
 * max_window_size. If yes reset it to beginning i.e. zero
 *
 */
static void mbm_fifo_in(struct sample *bw_stat, u32 val)
{
	bw_stat->mbmfifo[bw_stat->fifoin] = val;
	if (++bw_stat->fifoin == mbm_window_size)
		bw_stat->fifoin = 0;
}

/*
 * rmid_read_mbm checks whether it is LOCAL or Total MBM event and reads
 * its MSR counter. Check whether overflow occurred and handle it. Calculate
 * current bandwidth and updates its running average.
 *
 * MBM Counter Overflow:
 * Calculation of Current Bandwidth value:
 * If MSR is read within last 100ms, then we rturn the previous value
 * Currently perf receommends keeping 100ms between samples. Driver uses
 * this guideline. If the MSR was Read with in last 100ms, why  incur an
 * extra overhead of doing the MSR reads again.
 *
 * Bandwidth is calculated as:
 * memory bandwidth = (difference of  two msr counter values )/time difference
 *
 * cum_avg = Running Average of bandwidth with last 'n' bandwidth values of
 * the samples that are processed
 *
 * Sliding window is used to save the last 'n' samples. Where,
 * n = sliding_window_size and results in sliding window duration of 'n' secs.
 * The sliding window size by default set to
 * MBM_FIFO_SIZE_MIN. User can configure it to the values in the range
 * (MBM_FIFO_SIZE_MIN,MBM_FIFO_SIZE_MAX). The range for sliding window
 * is chosen based on a general criteria for monitoring duration. Example
 * for a short lived application, 10sec monitoring period gives
 * good characterization of its bandwidth consumption. For an application
 * that runs for longer duration, 300sec monitoring period gives better
 * characterization of its bandwidth consumption. Since the running average
 * calculated for total monitoring period, user gets the most accurate
 * average bandwidth for each monitoring period.
 *
 * Conversion from Bytes/sec to MB/sec:
 * current sample's  bandwidth is calculated in Bytes/sec.
 * Perf user space gets the values in units as specified by .scale and .unit
 * atrributes for the MBM event.
 */
static u64 rmid_read_mbm(unsigned int rmid, enum mbm_evt_type evt_type)
{
	u64  val, currentmsr, diff_time,  currentbw, bytes, prevavg;
	bool overflow = false, first = false;
	ktime_t cur_time;
	u32 eventid, index;
	struct sample *mbm_current;
	u32 vrmid = rmid_2_index(rmid);

	cur_time = ktime_get();
	if (evt_type & QOS_MBM_LOCAL_EVENT_MASK) {
		mbm_current = &mbm_local[vrmid];
		eventid     =  QOS_MBM_LOCAL_EVENT_ID;
	} else {
		mbm_current = &mbm_total[vrmid];
		eventid     = QOS_MBM_TOTAL_EVENT_ID;
	}

	prevavg = mbm_current->runavg;
	if (mbm_current->fifoin > 0)
		currentbw = mbm_current->mbmfifo[mbm_current->fifoin-1];
	else
		currentbw = prevavg;
	diff_time = ktime_ms_delta(cur_time,
				   mbm_current->prev_time);
	if (diff_time > MBM_TIME_DELTA_MIN) {

		wrmsr(MSR_IA32_QM_EVTSEL, eventid, rmid);
		rdmsrl(MSR_IA32_QM_CTR, val);

		if (val & (RMID_VAL_ERROR | RMID_VAL_UNAVAIL))
			return val;

		bytes = mbm_current->bytes;
		currentmsr = val;
		val &= MBM_CNTR_MAX;
		/* if MSR current read value is less than MSR previous read
		 * value then it is an overflow. MSR values are increasing
		 * when bandwidth consumption for the thread is non-zero;
		 * Overflow occurs, When MBM counter value reaches its
		 * maximum i.e. MBM_CNTR_MAX.
		 *
		 * After overflow, MSR current value goes back to zero and
		 * starts increasing again at the rate of bandwidth.
		 *
		 * Overflow handling:
		 * First overflow is detected by comparing current msr values
		 * will with the last read value. If current msr value is less
		 * than previous value then it is an overflow. When overflow
		 * occurs, (MBM_CONTR_MAX - prev msr value) is added the current
		 * msr value to the get actual value.
		 */

		if (val < bytes) {
			val = MBM_CNTR_MAX - bytes + val + 1;
			overflow = true;
		} else
			val = val - bytes;

		/*
		 * MBM_TIME_DELTA_EXP is picked as per MBM specs. As per
		 * hardware functionality, overflow can occur maximum once in a
		 * second. So latest we want to read the MSR counters is 1000ms.
		 * Minimum time interval between two MSR reads is 100ms. If
		 * read_rmid_mbm is called with in less than 100ms, use the
		 * previous sammple since perf also recommends to use the
		 * minimum sampling period of 100ms.
		 */

		if ((diff_time > MBM_TIME_DELTA_EXP) && (!prevavg))
		/* First sample, we can't use the time delta */
			first = true;

		if ((diff_time <= (MBM_TIME_DELTA_EXP + MBM_TIME_DELTA_MIN))  ||
			   overflow || first) {
			int averagebw, bwsum;

			/*
			 * For the first 'mbm_window_size -1' samples
			 * calculate average by adding the current sample's
			 * bandwidth to the sum of existing bandwidth values and
			 * dividing the sum with the #samples profiled so far
			*/
			averagebw = 0;
			index = mbm_current->index;
			currentbw =  (val * MSEC_PER_SEC) / diff_time;
			averagebw = currentbw;
			if (index    && (index < mbm_window_size)) {
				averagebw = prevavg  + currentbw / index -
				    prevavg / index;
			} else  if (index >= mbm_window_size) {
				/*
				 * Compute the sum of bandwidth for recent n-1
				 * sampland slide the window by 1
				 */
				bwsum = __mbm_fifo_sum_lastn_out(mbm_current);
				/*
				 * recalculate the running average by adding
				 * current bandwidth  and
				 * __mbm_fifo_sum_lastn_out which is the sum of
				 * last bandwidth values from the sliding
				 * window. The sum divided by mbm_window_size'
				 * is the new running average of the MBM
				 * Bandwidth
				 */
				averagebw = (bwsum + currentbw) /
					     mbm_window_size;
			}

			/* save the current sample's bandwidth in fifo */
			mbm_fifo_in(mbm_current, currentbw);
			mbm_current->index++;
			mbm_current->runavg = averagebw;
			mbm_current->bytes = currentmsr;
			mbm_current->prev_time = cur_time;

		}
	}
	/* No change, return the existing running average */
	if (evt_type & QOS_MBM_AVG_EVENT_MASK)
		return mbm_current->runavg;
	else
		return currentbw;
}

static void intel_mbm_event_update(struct perf_event *event)
{
	unsigned int rmid;
	u64 val = 0;

	/*
	 * Task events are handled by intel_cqm_event_count().
	 */

	rmid = event->hw.cqm_rmid;
	if (!__rmid_valid(rmid))
		return;
	val = rmid_read_mbm(rmid, event->attr.config);
	/*
	 * Ignore this reading on error states and do not update the value.
	 */
	if (val & (RMID_VAL_ERROR | RMID_VAL_UNAVAIL))
		return;

	local64_set(&event->count, val);
}

/*
 * Initially use this constant for both the limbo queue time and the
 * rotation timer interval, pmu::hrtimer_interval_ms.
 *
 * They don't need to be the same, but the two are related since if you
 * rotate faster than you recycle RMIDs, you may run out of available
 * RMIDs.
 */
#define RMID_DEFAULT_QUEUE_TIME 250	/* ms */

static unsigned int __rmid_queue_time_ms = RMID_DEFAULT_QUEUE_TIME;

/*
 * intel_cqm_rmid_stabilize - move RMIDs from limbo to free list
 * @nr_available: number of freeable RMIDs on the limbo list
 *
 * Quiescent state; wait for all 'freed' RMIDs to become unused, i.e. no
 * cachelines are tagged with those RMIDs. After this we can reuse them
 * and know that the current set of active RMIDs is stable.
 *
 * Return %true or %false depending on whether stabilization needs to be
 * reattempted.
 *
 * If we return %true then @nr_available is updated to indicate the
 * number of RMIDs on the limbo list that have been queued for the
 * minimum queue time (RMID_AVAILABLE), but whose data occupancy values
 * are above __intel_cqm_threshold.
 */
static bool intel_cqm_rmid_stabilize(unsigned int *available)
{
	struct cqm_rmid_entry *entry, *tmp;

	lockdep_assert_held(&cache_mutex);

	*available = 0;
	list_for_each_entry(entry, &cqm_rmid_limbo_lru, list) {
		unsigned long min_queue_time;
		unsigned long now = jiffies;

		/*
		 * We hold RMIDs placed into limbo for a minimum queue
		 * time. Before the minimum queue time has elapsed we do
		 * not recycle RMIDs.
		 *
		 * The reasoning is that until a sufficient time has
		 * passed since we stopped using an RMID, any RMID
		 * placed onto the limbo list will likely still have
		 * data tagged in the cache, which means we'll probably
		 * fail to recycle it anyway.
		 *
		 * We can save ourselves an expensive IPI by skipping
		 * any RMIDs that have not been queued for the minimum
		 * time.
		 */
		min_queue_time = entry->queue_time +
			msecs_to_jiffies(__rmid_queue_time_ms);

		if (time_after(min_queue_time, now))
			break;

		entry->state = RMID_AVAILABLE;
		(*available)++;
	}

	/*
	 * Fast return if none of the RMIDs on the limbo list have been
	 * sitting on the queue for the minimum queue time.
	 */
	if (!*available)
		return false;

	/*
	 * Test whether an RMID is free for each package.
	 */
	on_each_cpu_mask(&cqm_cpumask, intel_cqm_stable, NULL, true);

	list_for_each_entry_safe(entry, tmp, &cqm_rmid_limbo_lru, list) {
		/*
		 * Exhausted all RMIDs that have waited min queue time.
		 */
		if (entry->state == RMID_YOUNG)
			break;

		if (entry->state == RMID_DIRTY)
			continue;

		list_del(&entry->list);	/* remove from limbo */
		intel_cqm_free_rmid(entry);
	}


	return __rmid_valid(intel_cqm_rotation_rmid);
}

/*
 * Pick a victim group and move it to the tail of the group list.
 * @next: The first group without an RMID
 */
static void __intel_cqm_pick_and_rotate(struct perf_event *next)
{
	struct perf_event *rotor;
	u32 rmid;

	lockdep_assert_held(&cache_mutex);

	rotor = list_first_entry(&cache_groups, struct perf_event,
				 hw.cqm_groups_entry);

	/*
	 * The group at the front of the list should always have a valid
	 * RMID. If it doesn't then no groups have RMIDs assigned and we
	 * don't need to rotate the list.
	 */
	if (next == rotor)
		return;

	rmid = intel_cqm_xchg_rmid(rotor, INVALID_RMID);
	__put_rmid(rmid);

	list_rotate_left(&cache_groups);
}

/*
 * Deallocate the RMIDs from any events that conflict with @event, and
 * place them on the back of the group list.
 */
static void intel_cqm_sched_out_conflicting_events(struct perf_event *event)
{
	struct perf_event *group, *g;
	u32 rmid;

	lockdep_assert_held(&cache_mutex);

	list_for_each_entry_safe(group, g, &cache_groups, hw.cqm_groups_entry) {
		if (group == event)
			continue;

		rmid = group->hw.cqm_rmid;

		/*
		 * Skip events that don't have a valid RMID.
		 */
		if (!__rmid_valid(rmid))
			continue;

		/*
		 * No conflict? No problem! Leave the event alone.
		 */
		if (!__conflict_event(group, event))
			continue;

		intel_cqm_xchg_rmid(group, INVALID_RMID);
		__put_rmid(rmid);
	}
}

/*
 * Attempt to rotate the groups and assign new RMIDs.
 *
 * We rotate for two reasons,
 *   1. To handle the scheduling of conflicting events
 *   2. To recycle RMIDs
 *
 * Rotating RMIDs is complicated because the hardware doesn't give us
 * any clues.
 *
 * There's problems with the hardware interface; when you change the
 * task:RMID map cachelines retain their 'old' tags, giving a skewed
 * picture. In order to work around this, we must always keep one free
 * RMID - intel_cqm_rotation_rmid.
 *
 * Rotation works by taking away an RMID from a group (the old RMID),
 * and assigning the free RMID to another group (the new RMID). We must
 * then wait for the old RMID to not be used (no cachelines tagged).
 * This ensure that all cachelines are tagged with 'active' RMIDs. At
 * this point we can start reading values for the new RMID and treat the
 * old RMID as the free RMID for the next rotation.
 *
 * Return %true or %false depending on whether we did any rotating.
 */
static bool __intel_cqm_rmid_rotate(void)
{
	struct perf_event *group, *start = NULL;
	unsigned int threshold_limit;
	unsigned int nr_needed = 0;
	unsigned int nr_available;
	bool rotated = false;

	mutex_lock(&cache_mutex);

again:
	/*
	 * Fast path through this function if there are no groups and no
	 * RMIDs that need cleaning.
	 */
	if (list_empty(&cache_groups) && list_empty(&cqm_rmid_limbo_lru))
		goto out;

	list_for_each_entry(group, &cache_groups, hw.cqm_groups_entry) {
		if (!__rmid_valid(group->hw.cqm_rmid)) {
			if (!start)
				start = group;
			nr_needed++;
		}
	}

	/*
	 * We have some event groups, but they all have RMIDs assigned
	 * and no RMIDs need cleaning.
	 */
	if (!nr_needed && list_empty(&cqm_rmid_limbo_lru))
		goto out;

	if (!nr_needed)
		goto stabilize;

	/*
	 * We have more event groups without RMIDs than available RMIDs,
	 * or we have event groups that conflict with the ones currently
	 * scheduled.
	 *
	 * We force deallocate the rmid of the group at the head of
	 * cache_groups. The first event group without an RMID then gets
	 * assigned intel_cqm_rotation_rmid. This ensures we always make
	 * forward progress.
	 *
	 * Rotate the cache_groups list so the previous head is now the
	 * tail.
	 */
	__intel_cqm_pick_and_rotate(start);

	/*
	 * If the rotation is going to succeed, reduce the threshold so
	 * that we don't needlessly reuse dirty RMIDs.
	 */
	if (__rmid_valid(intel_cqm_rotation_rmid)) {
		intel_cqm_xchg_rmid(start, intel_cqm_rotation_rmid);
		intel_cqm_rotation_rmid = __get_rmid();

		intel_cqm_sched_out_conflicting_events(start);

		if (__intel_cqm_threshold)
			__intel_cqm_threshold--;
	}

	rotated = true;

stabilize:
	/*
	 * We now need to stablize the RMID we freed above (if any) to
	 * ensure that the next time we rotate we have an RMID with zero
	 * occupancy value.
	 *
	 * Alternatively, if we didn't need to perform any rotation,
	 * we'll have a bunch of RMIDs in limbo that need stabilizing.
	 */
	threshold_limit = __intel_cqm_max_threshold / cqm_l3_scale;

	while (intel_cqm_rmid_stabilize(&nr_available) &&
	       __intel_cqm_threshold < threshold_limit) {
		unsigned int steal_limit;

		/*
		 * Don't spin if nobody is actively waiting for an RMID,
		 * the rotation worker will be kicked as soon as an
		 * event needs an RMID anyway.
		 */
		if (!nr_needed)
			break;

		/* Allow max 25% of RMIDs to be in limbo. */
		steal_limit = (cqm_max_rmid + 1) / 4;

		/*
		 * We failed to stabilize any RMIDs so our rotation
		 * logic is now stuck. In order to make forward progress
		 * we have a few options:
		 *
		 *   1. rotate ("steal") another RMID
		 *   2. increase the threshold
		 *   3. do nothing
		 *
		 * We do both of 1. and 2. until we hit the steal limit.
		 *
		 * The steal limit prevents all RMIDs ending up on the
		 * limbo list. This can happen if every RMID has a
		 * non-zero occupancy above threshold_limit, and the
		 * occupancy values aren't dropping fast enough.
		 *
		 * Note that there is prioritisation at work here - we'd
		 * rather increase the number of RMIDs on the limbo list
		 * than increase the threshold, because increasing the
		 * threshold skews the event data (because we reuse
		 * dirty RMIDs) - threshold bumps are a last resort.
		 */
		if (nr_available < steal_limit)
			goto again;

		__intel_cqm_threshold++;
	}

out:
	mutex_unlock(&cache_mutex);
	return rotated;
}

static void intel_cqm_rmid_rotate(struct work_struct *work);

static DECLARE_DELAYED_WORK(intel_cqm_rmid_work, intel_cqm_rmid_rotate);

static struct pmu intel_cqm_pmu;

static void intel_cqm_rmid_rotate(struct work_struct *work)
{
	unsigned long delay;

	__intel_cqm_rmid_rotate();

	delay = msecs_to_jiffies(intel_cqm_pmu.hrtimer_interval_ms);
	schedule_delayed_work(&intel_cqm_rmid_work, delay);
}

/*
 * Find a group and setup RMID.
 *
 * If we're part of a group, we use the group's RMID.
 */
static void intel_cqm_setup_event(struct perf_event *event,
				  struct perf_event **group)
{
	struct perf_event *iter;
	bool conflict = false;
	u32 rmid;

	list_for_each_entry(iter, &cache_groups, hw.cqm_groups_entry) {
		rmid = iter->hw.cqm_rmid;

		if (__match_event(iter, event)) {
			/* All tasks in a group share an RMID */
			event->hw.cqm_rmid = rmid;
			*group = iter;
			return;
		}

		/*
		 * We only care about conflicts for events that are
		 * actually scheduled in (and hence have a valid RMID).
		 */
		if (__conflict_event(iter, event) && __rmid_valid(rmid))
			conflict = true;
	}

	if (conflict)
		rmid = INVALID_RMID;
	else
		rmid = __get_rmid();

	if (event->attr.config == QOS_L3_OCCUP_EVENT_ID) {
		struct cqm_rmid_entry *entry;

		entry = __rmid_entry(rmid);
		entry->is_cqm = true;
	}

	event->hw.cqm_rmid = rmid;
	if ((event->attr.config >= QOS_MBM_TOTAL_EVENT_ID) && (event->attr.config <= QOS_MBM_LOCAL_AVG_EVENT_ID))
		rmid_read_mbm(rmid, event->attr.config);
}

static void intel_cqm_event_read(struct perf_event *event)
{
	unsigned long flags;
	u32 rmid;
	u64 val;

	/*
	 * Task events are handled by intel_cqm_event_count().
	 */
	if (event->cpu == -1)
		return;

	if  ((event->attr.config >= QOS_MBM_TOTAL_EVENT_ID) &&
	     (event->attr.config <= QOS_MBM_LOCAL_EVENT_ID))
		intel_mbm_event_update(event);

	if (event->attr.config !=  QOS_L3_OCCUP_EVENT_ID)
		return;

	raw_spin_lock_irqsave(&cache_lock, flags);
	rmid = event->hw.cqm_rmid;

	if (!__rmid_valid(rmid))
		goto out;

	val = __rmid_read(rmid);

	/*
	 * Ignore this reading on error states and do not update the value.
	 */
	if (val & (RMID_VAL_ERROR | RMID_VAL_UNAVAIL))
		goto out;

	local64_set(&event->count, val);
out:
	raw_spin_unlock_irqrestore(&cache_lock, flags);
}

static void __intel_cqm_event_count(void *info)
{
	struct rmid_read *rr = info;
	u64 val;

	val = __rmid_read(rr->rmid);

	if (val & (RMID_VAL_ERROR | RMID_VAL_UNAVAIL))
		return;

	atomic64_add(val, &rr->value);
}

static inline bool cqm_group_leader(struct perf_event *event)
{
	return !list_empty(&event->hw.cqm_groups_entry);
}

static void mbm_stop_hrtimer(struct mbm_pmu *pmu)
{
	hrtimer_cancel(&pmu->hrtimer);
}

static void __intel_mbm_event_count(void *info)
{
	struct rmid_read *rr = info;
	u64 val;

	val = rmid_read_mbm(rr->rmid, rr->evt_type);
	if (val & (RMID_VAL_ERROR | RMID_VAL_UNAVAIL))
		return;
	atomic64_add(val, &rr->value);
}

static u64 intel_mbm_event_count(struct perf_event *event, struct rmid_read *rr)
{
	struct mbm_pmu *pmu = __this_cpu_read(mbm_pmu);

	on_each_cpu_mask(&cqm_cpumask, __intel_mbm_event_count,
		   rr, 1);
	if (pmu) {
		pmu->n_active--;
		if (pmu->n_active == 0)
			mbm_stop_hrtimer(pmu);
	}
	if (event->hw.cqm_rmid == rr->rmid)
		local64_set(&event->count, atomic64_read(&rr->value));
	return __perf_event_count(event);

}
static u64 intel_cqm_event_count(struct perf_event *event)
{
	unsigned long flags;
	struct rmid_read rr = {
		.value = ATOMIC64_INIT(0),
	};

	/*
	 * We only need to worry about task events. System-wide events
	 * are handled like usual, i.e. entirely with
	 * intel_cqm_event_read().
	 */
	if (event->cpu != -1)
		return __perf_event_count(event);

	/*
	 * Only the group leader gets to report values. This stops us
	 * reporting duplicate values to userspace, and gives us a clear
	 * rule for which task gets to report the values.
	 *
	 * Note that it is impossible to attribute these values to
	 * specific packages - we forfeit that ability when we create
	 * task events.
	 */
	if (!cqm_group_leader(event)) {
		struct cqm_rmid_entry *entry;

		entry = __rmid_entry(event->hw.cqm_rmid);
		if (!entry->is_multi_event)
			return 0;
	}

	/*
	 * Getting up-to-date values requires an SMP IPI which is not
	 * possible if we're being called in interrupt context. Return
	 * the cached values instead.
	 */
	if (unlikely(in_interrupt()))
		goto out;

	/*
	 * Notice that we don't perform the reading of an RMID
	 * atomically, because we can't hold a spin lock across the
	 * IPIs.
	 *
	 * Speculatively perform the read, since @event might be
	 * assigned a different (possibly invalid) RMID while we're
	 * busying performing the IPI calls. It's therefore necessary to
	 * check @event's RMID afterwards, and if it has changed,
	 * discard the result of the read.
	 */
	rr.rmid = ACCESS_ONCE(event->hw.cqm_rmid);

	if (!__rmid_valid(rr.rmid))
		goto out;

	if (event->attr.config == QOS_L3_OCCUP_EVENT_ID)
		on_each_cpu_mask(&cqm_cpumask, __intel_cqm_event_count, &rr, 1);

	if (((event->attr.config >= QOS_MBM_TOTAL_EVENT_ID) &&
	     (event->attr.config <= QOS_MBM_LOCAL_AVG_EVENT_ID))  && (is_mbm)) {
		rr.evt_type = event->attr.config;
		return intel_mbm_event_count(event, &rr);
	}
	raw_spin_lock_irqsave(&cache_lock, flags);
	if (event->hw.cqm_rmid == rr.rmid)
		local64_set(&event->count, atomic64_read(&rr.value));
	raw_spin_unlock_irqrestore(&cache_lock, flags);
out:
	return __perf_event_count(event);
}

static void mbm_start_hrtimer(struct mbm_pmu *pmu)
{
	hrtimer_start_range_ns(&(pmu->hrtimer),
				 pmu->timer_interval, 0,
				 HRTIMER_MODE_REL_PINNED);
}

static enum hrtimer_restart mbm_hrtimer_handle(struct hrtimer *hrtimer)
{
	struct mbm_pmu *pmu = __this_cpu_read(mbm_pmu);
	struct perf_event *event;

	if (!pmu->n_active)
		return HRTIMER_NORESTART;
	list_for_each_entry(event, &pmu->active_list, active_entry)
		intel_mbm_event_update(event);
	hrtimer_forward_now(hrtimer, pmu->timer_interval);
	return HRTIMER_RESTART;
}

static void mbm_hrtimer_init(struct mbm_pmu *pmu)
{
	struct hrtimer *hr = &pmu->hrtimer;

	hrtimer_init(hr, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
	hr->function = mbm_hrtimer_handle;
}

static void intel_mbm_event_start(struct perf_event *event, int mode)
{

	if (((event->attr.config >= QOS_MBM_TOTAL_EVENT_ID) &&
	     (event->attr.config <= QOS_MBM_LOCAL_EVENT_ID))  && (is_mbm)) {
		struct mbm_pmu *pmu = __this_cpu_read(mbm_pmu);

		if (pmu) {
			pmu->n_active++;
			list_add_tail(&event->active_entry,
				      &pmu->active_list);
			if (pmu->n_active == 1)
				mbm_start_hrtimer(pmu);
		}
	}

}

static void intel_cqm_event_start(struct perf_event *event, int mode)
{
	struct intel_pqr_state *state = this_cpu_ptr(&pqr_state);
	u32 rmid = event->hw.cqm_rmid;

	if (!(event->hw.cqm_state & PERF_HES_STOPPED))
		return;

	event->hw.cqm_state &= ~PERF_HES_STOPPED;

	if (state->rmid_usecnt++) {
		if (!WARN_ON_ONCE(state->rmid != rmid))
			return;
	} else {
		WARN_ON_ONCE(state->rmid);
	}

	state->rmid = rmid;
	wrmsr(MSR_IA32_PQR_ASSOC, rmid, state->closid);
	intel_mbm_event_start(event, mode);

}

static void intel_mbm_event_stop(struct perf_event *event, int mode)
{
	struct mbm_pmu *pmu = __this_cpu_read(mbm_pmu);

	if (pmu) {
		intel_mbm_event_update(event);
		if ((pmu->n_active >  0) && (event->cpu != -1))
			pmu->n_active--;
			if (pmu->n_active == 0)
				mbm_stop_hrtimer(pmu);
		if (!list_empty(&event->active_entry))
			list_del(&event->active_entry);
	}

}

static void intel_cqm_event_stop(struct perf_event *event, int mode)
{
	struct intel_pqr_state *state = this_cpu_ptr(&pqr_state);

	if (event->hw.cqm_state & PERF_HES_STOPPED)
		return;

	event->hw.cqm_state |= PERF_HES_STOPPED;

	if (event->attr.config == QOS_L3_OCCUP_EVENT_ID)
		intel_cqm_event_read(event);

	if ((event->attr.config >= QOS_MBM_TOTAL_EVENT_ID) &&
	    (event->attr.config <= QOS_MBM_LOCAL_EVENT_ID))
		intel_mbm_event_update(event);

	if (!--state->rmid_usecnt) {
		state->rmid = 0;
		wrmsr(MSR_IA32_PQR_ASSOC, 0, state->closid);
	} else {
		WARN_ON_ONCE(!state->rmid);
	}

	intel_mbm_event_stop(event, mode);
}

static int intel_cqm_event_add(struct perf_event *event, int mode)
{
	unsigned long flags;
	u32 rmid;

	raw_spin_lock_irqsave(&cache_lock, flags);

	event->hw.cqm_state = PERF_HES_STOPPED;
	rmid = event->hw.cqm_rmid;

	if (__rmid_valid(rmid) && (mode & PERF_EF_START))
		intel_cqm_event_start(event, mode);

	raw_spin_unlock_irqrestore(&cache_lock, flags);

	return 0;
}

static void intel_cqm_event_destroy(struct perf_event *event)
{
	struct perf_event *group_other = NULL;

	mutex_lock(&cache_mutex);

	/*
	 * If there's another event in this group...
	 */
	if (!list_empty(&event->hw.cqm_group_entry)) {
		group_other = list_first_entry(&event->hw.cqm_group_entry,
					       struct perf_event,
					       hw.cqm_group_entry);
		list_del(&event->hw.cqm_group_entry);
	}

	/*
	 * And we're the group leader..
	 */
	if (cqm_group_leader(event)) {
		/*
		 * If there was a group_other, make that leader, otherwise
		 * destroy the group and return the RMID.
		 */
		if (group_other) {
			list_replace(&event->hw.cqm_groups_entry,
				     &group_other->hw.cqm_groups_entry);
		} else {
			u32 rmid = event->hw.cqm_rmid;

			if (__rmid_valid(rmid))
				__put_rmid(rmid);
			list_del(&event->hw.cqm_groups_entry);
		}
	}

	mutex_unlock(&cache_mutex);
}

static int intel_cqm_event_init(struct perf_event *event)
{
	struct perf_event *group = NULL;
	bool rotate = false;

	if (event->attr.type != intel_cqm_pmu.type)
		return -ENOENT;

	if ((event->attr.config < QOS_L3_OCCUP_EVENT_ID) ||
	     (event->attr.config > QOS_MBM_LOCAL_AVG_EVENT_ID))
		return -EINVAL;

	/* unsupported modes and filters */
	if (event->attr.exclude_user   ||
	    event->attr.exclude_kernel ||
	    event->attr.exclude_hv     ||
	    event->attr.exclude_idle   ||
	    event->attr.exclude_host   ||
	    event->attr.exclude_guest  ||
	    event->attr.sample_period) /* no sampling */
		return -EINVAL;

	INIT_LIST_HEAD(&event->hw.cqm_group_entry);
	INIT_LIST_HEAD(&event->hw.cqm_groups_entry);

	event->destroy = intel_cqm_event_destroy;

	mutex_lock(&cache_mutex);

	/* Will also set rmid */
	intel_cqm_setup_event(event, &group);

	if (group) {
		list_add_tail(&event->hw.cqm_group_entry,
			      &group->hw.cqm_group_entry);
	} else {
		list_add_tail(&event->hw.cqm_groups_entry,
			      &cache_groups);

		/*
		 * All RMIDs are either in use or have recently been
		 * used. Kick the rotation worker to clean/free some.
		 *
		 * We only do this for the group leader, rather than for
		 * every event in a group to save on needless work.
		 */
		if (!__rmid_valid(event->hw.cqm_rmid))
			rotate = true;
	}

	mutex_unlock(&cache_mutex);

	if (rotate)
		schedule_delayed_work(&intel_cqm_rmid_work, 0);

	return 0;
}

EVENT_ATTR_STR(llc_occupancy, intel_cqm_llc, "event=0x01");
EVENT_ATTR_STR(llc_occupancy.per-pkg, intel_cqm_llc_pkg, "1");
EVENT_ATTR_STR(llc_occupancy.unit, intel_cqm_llc_unit, "Bytes");
EVENT_ATTR_STR(llc_occupancy.scale, intel_cqm_llc_scale, NULL);
EVENT_ATTR_STR(llc_occupancy.snapshot, intel_cqm_llc_snapshot, "1");

EVENT_ATTR_STR(total_bw, intel_cqm_total_bw, "event=0x02");
EVENT_ATTR_STR(total_bw.per-pkg, intel_cqm_total_bw_pkg, "1");
EVENT_ATTR_STR(total_bw.unit, intel_cqm_total_bw_unit, "MB/sec");
EVENT_ATTR_STR(total_bw.scale, intel_cqm_total_bw_scale, NULL);
EVENT_ATTR_STR(total_bw.snapshot, intel_cqm_total_bw_snapshot, "1");
EVENT_ATTR_STR(total_bw.runavg_nosamples,
				intel_cqm_total_bw_runavg_nosamples, "10");
EVENT_ATTR_STR(local_bw.runavg_nosamples,
				intel_cqm_local_bw_runavg_nosamples, "10");


EVENT_ATTR_STR(local_bw, intel_cqm_local_bw, "event=0x03");
EVENT_ATTR_STR(local_bw.per-pkg, intel_cqm_local_bw_pkg, "1");
EVENT_ATTR_STR(local_bw.unit, intel_cqm_local_bw_unit, "MB/sec");
EVENT_ATTR_STR(local_bw.scale, intel_cqm_local_bw_scale, NULL);
EVENT_ATTR_STR(local_bw.snapshot, intel_cqm_local_bw_snapshot, "1");

EVENT_ATTR_STR(avg_total_bw, intel_cqm_avg_total_bw, "event=0x04");
EVENT_ATTR_STR(avg_total_bw.per-pkg, intel_cqm_avg_total_bw_pkg, "1");
EVENT_ATTR_STR(avg_total_bw.unit, intel_cqm_avg_total_bw_unit, "MB/sec");
EVENT_ATTR_STR(avg_total_bw.scale, intel_cqm_avg_total_bw_scale, NULL);
EVENT_ATTR_STR(avg_total_bw.snapshot, intel_cqm_avg_total_bw_snapshot, "1");

EVENT_ATTR_STR(avg_local_bw, intel_cqm_avg_local_bw, "event=0x05");
EVENT_ATTR_STR(avg_local_bw.per-pkg, intel_cqm_avg_local_bw_pkg, "1");
EVENT_ATTR_STR(avg_local_bw.unit, intel_cqm_avg_local_bw_unit, "MB/sec");
EVENT_ATTR_STR(avg_local_bw.scale, intel_cqm_avg_local_bw_scale, NULL);
EVENT_ATTR_STR(avg_local_bw.snapshot, intel_cqm_avg_local_bw_snapshot, "1");

static struct attribute *intel_cqm_events_attr[] = {
	EVENT_PTR(intel_cqm_llc),
	EVENT_PTR(intel_cqm_llc_pkg),
	EVENT_PTR(intel_cqm_llc_unit),
	EVENT_PTR(intel_cqm_llc_scale),
	EVENT_PTR(intel_cqm_llc_snapshot),
	NULL,
};

static struct attribute *intel_mbm_events_attr[] = {
	EVENT_PTR(intel_cqm_total_bw),
	EVENT_PTR(intel_cqm_local_bw),
	EVENT_PTR(intel_cqm_avg_total_bw),
	EVENT_PTR(intel_cqm_avg_local_bw),
	EVENT_PTR(intel_cqm_total_bw_pkg),
	EVENT_PTR(intel_cqm_local_bw_pkg),
	EVENT_PTR(intel_cqm_avg_total_bw_pkg),
	EVENT_PTR(intel_cqm_avg_local_bw_pkg),
	EVENT_PTR(intel_cqm_total_bw_unit),
	EVENT_PTR(intel_cqm_local_bw_unit),
	EVENT_PTR(intel_cqm_avg_total_bw_unit),
	EVENT_PTR(intel_cqm_avg_local_bw_unit),
	EVENT_PTR(intel_cqm_total_bw_scale),
	EVENT_PTR(intel_cqm_local_bw_scale),
	EVENT_PTR(intel_cqm_avg_total_bw_scale),
	EVENT_PTR(intel_cqm_avg_local_bw_scale),
	EVENT_PTR(intel_cqm_total_bw_snapshot),
	EVENT_PTR(intel_cqm_local_bw_snapshot),
	EVENT_PTR(intel_cqm_avg_total_bw_snapshot),
	EVENT_PTR(intel_cqm_avg_local_bw_snapshot),
	EVENT_PTR(intel_cqm_total_bw_runavg_nosamples),
	EVENT_PTR(intel_cqm_local_bw_runavg_nosamples),
	NULL,
};

static struct attribute *intel_cmt_mbm_events_attr[] = {
	EVENT_PTR(intel_cqm_llc),
	EVENT_PTR(intel_cqm_total_bw),
	EVENT_PTR(intel_cqm_local_bw),
	EVENT_PTR(intel_cqm_avg_total_bw),
	EVENT_PTR(intel_cqm_avg_local_bw),
	EVENT_PTR(intel_cqm_llc_pkg),
	EVENT_PTR(intel_cqm_total_bw_pkg),
	EVENT_PTR(intel_cqm_local_bw_pkg),
	EVENT_PTR(intel_cqm_avg_total_bw_pkg),
	EVENT_PTR(intel_cqm_avg_local_bw_pkg),
	EVENT_PTR(intel_cqm_llc_unit),
	EVENT_PTR(intel_cqm_total_bw_unit),
	EVENT_PTR(intel_cqm_local_bw_unit),
	EVENT_PTR(intel_cqm_avg_total_bw_unit),
	EVENT_PTR(intel_cqm_avg_local_bw_unit),
	EVENT_PTR(intel_cqm_llc_scale),
	EVENT_PTR(intel_cqm_total_bw_scale),
	EVENT_PTR(intel_cqm_local_bw_scale),
	EVENT_PTR(intel_cqm_avg_total_bw_scale),
	EVENT_PTR(intel_cqm_avg_local_bw_scale),
	EVENT_PTR(intel_cqm_llc_snapshot),
	EVENT_PTR(intel_cqm_total_bw_snapshot),
	EVENT_PTR(intel_cqm_local_bw_snapshot),
	EVENT_PTR(intel_cqm_avg_total_bw_snapshot),
	EVENT_PTR(intel_cqm_avg_local_bw_snapshot),
	EVENT_PTR(intel_cqm_total_bw_runavg_nosamples),
	EVENT_PTR(intel_cqm_local_bw_runavg_nosamples),
	NULL,
};

static struct attribute_group intel_cqm_events_group = {
	.name = "events",
	.attrs = NULL,
};

PMU_FORMAT_ATTR(event, "config:0-7");
static struct attribute *intel_cqm_formats_attr[] = {
	&format_attr_event.attr,
	NULL,
};

static struct attribute_group intel_cqm_format_group = {
	.name = "format",
	.attrs = intel_cqm_formats_attr,
};

static ssize_t
max_recycle_threshold_show(struct device *dev, struct device_attribute *attr,
			   char *page)
{
	ssize_t rv;

	mutex_lock(&cache_mutex);
	rv = snprintf(page, PAGE_SIZE-1, "%u\n", __intel_cqm_max_threshold);
	mutex_unlock(&cache_mutex);

	return rv;
}

static ssize_t
sliding_window_size_show(struct device *dev, struct device_attribute *attr,
		char *page)
{
	ssize_t rv;

	rv = snprintf(page, PAGE_SIZE-1, "%u\n", mbm_window_size);
	return rv;
}

static ssize_t
max_recycle_threshold_store(struct device *dev,
			    struct device_attribute *attr,
			    const char *buf, size_t count)
{
	unsigned int bytes, cachelines;
	int ret;

	ret = kstrtouint(buf, 0, &bytes);
	if (ret)
		return ret;

	mutex_lock(&cache_mutex);

	__intel_cqm_max_threshold = bytes;
	cachelines = bytes / cqm_l3_scale;

	/*
	 * The new maximum takes effect immediately.
	 */
	if (__intel_cqm_threshold > cachelines)
		__intel_cqm_threshold = cachelines;

	mutex_unlock(&cache_mutex);

	return count;
}

static ssize_t
sliding_window_size_store(struct device *dev,
			  struct device_attribute *attr,
			  const char *buf, size_t count)
{
	unsigned int bytes;
	int ret;

	ret = kstrtouint(buf, 0, &bytes);
	if (ret)
		return ret;

	mutex_lock(&cache_mutex);
	if (bytes >= MBM_FIFO_SIZE_MIN && bytes <= MBM_FIFO_SIZE_MAX)
		mbm_window_size = bytes;
	else {
		mutex_unlock(&cache_mutex);
		return -EINVAL;
	}
	mutex_unlock(&cache_mutex);

	return count;
}

static DEVICE_ATTR_RW(max_recycle_threshold);
static DEVICE_ATTR_RW(sliding_window_size);

static struct attribute *intel_cqm_attrs[] = {
	&dev_attr_max_recycle_threshold.attr,
	&dev_attr_sliding_window_size.attr,
	NULL,
};

static const struct attribute_group intel_cqm_group = {
	.attrs = intel_cqm_attrs,
};

static const struct attribute_group *intel_cqm_attr_groups[] = {
	&intel_cqm_events_group,
	&intel_cqm_format_group,
	&intel_cqm_group,
	NULL,
};

static struct pmu intel_cqm_pmu = {
	.hrtimer_interval_ms = RMID_DEFAULT_QUEUE_TIME,
	.attr_groups	     = intel_cqm_attr_groups,
	.task_ctx_nr	     = perf_sw_context,
	.event_init	     = intel_cqm_event_init,
	.add		     = intel_cqm_event_add,
	.del		     = intel_cqm_event_stop,
	.start		     = intel_cqm_event_start,
	.stop		     = intel_cqm_event_stop,
	.read		     = intel_cqm_event_read,
	.count		     = intel_cqm_event_count,
};

static inline void cqm_pick_event_reader(int cpu)
{
	int phys_id = topology_physical_package_id(cpu);
	int i;

	for_each_cpu(i, &cqm_cpumask) {
		if (phys_id == topology_physical_package_id(i))
			return;	/* already got reader for this socket */
	}

	cpumask_set_cpu(cpu, &cqm_cpumask);
}

static int intel_mbm_cpu_prepare(unsigned int cpu)
{
	struct mbm_pmu *pmu = per_cpu(mbm_pmu, cpu);

	if ((!pmu) && (is_mbm)) {
		pmu = kzalloc_node(sizeof(*mbm_pmu), GFP_KERNEL, NUMA_NO_NODE);
		if (!pmu)
			return  -ENOMEM;
		INIT_LIST_HEAD(&pmu->active_list);
		pmu->pmu = &intel_cqm_pmu;
		pmu->timer_interval = ms_to_ktime(MBM_TIME_DELTA_EXP);
		per_cpu(mbm_pmu, cpu) = pmu;
		mbm_hrtimer_init(pmu);
	}
	return 0;
}

static int intel_cqm_cpu_starting(unsigned int cpu)
{
	struct intel_pqr_state *state = &per_cpu(pqr_state, cpu);
	struct cpuinfo_x86 *c = &cpu_data(cpu);

	state->rmid = 0;
	state->closid = 0;
	state->rmid_usecnt = 0;

	WARN_ON(c->x86_cache_max_rmid != cqm_max_rmid);
	WARN_ON(c->x86_cache_occ_scale != cqm_l3_scale);

	return intel_mbm_cpu_prepare(cpu);
}

static void intel_cqm_cpu_exit(unsigned int cpu)
{
	int phys_id = topology_physical_package_id(cpu);
	int i;
	struct mbm_pmu *pmu = per_cpu(mbm_pmu, cpu);

	/*
	 * Is @cpu a designated cqm reader?
	 */
	if (!cpumask_test_and_clear_cpu(cpu, &cqm_cpumask))
		return;

	for_each_online_cpu(i) {
		if (i == cpu)
			continue;

		if (phys_id == topology_physical_package_id(i)) {
			cpumask_set_cpu(i, &cqm_cpumask);
			break;
		}
	}

	/* cancel overflow polling timer for CPU */
	if (pmu)
		mbm_stop_hrtimer(pmu);

}

static int intel_cqm_cpu_notifier(struct notifier_block *nb,
				  unsigned long action, void *hcpu)
{
	unsigned int cpu  = (unsigned long)hcpu;
	int ret;

	switch (action & ~CPU_TASKS_FROZEN) {
	case CPU_DOWN_PREPARE:
		intel_cqm_cpu_exit(cpu);
		break;
	case CPU_STARTING:
		ret = intel_cqm_cpu_starting(cpu);
		if (ret)
			return ret;
		cqm_pick_event_reader(cpu);
		break;
	}

	return NOTIFY_OK;
}

static const struct x86_cpu_id intel_cqm_match[] = {
	{ .vendor = X86_VENDOR_INTEL, .feature = X86_FEATURE_CQM_OCCUP_LLC },
	{}
};

static const struct x86_cpu_id intel_mbm_match[] = {
	{ .vendor = X86_VENDOR_INTEL, .feature = X86_FEATURE_CQM_MBM_LOCAL },
	{}
};

static int  intel_mbm_init(void)
{
	u32 i;
	int ret, array_size;
	char scale[20], *str = NULL;

	if (!x86_match_cpu(intel_mbm_match))
		return -ENODEV;
	is_mbm = true;
	/*
	 * MBM counter values are  in Bytes. To convert this to MBytes:
	 * Bytes / 1.0e6 gives the MBytes.  Hardware uses upscale factor
	 * as given by cqm_l3_scale. Muliply upscale factor by 1/1.0e6
	 * to set the scale to get the perf output in MBytes/sec
	 */

	snprintf(scale, sizeof(scale), "%u%s", cqm_l3_scale, "e-6");
	str = kstrdup(scale, GFP_KERNEL);
	if (!str) {
		is_mbm = false;
		return -ENOMEM;
	}
	if (cqm_llc_occ)
		intel_cqm_events_group.attrs =
			  intel_cmt_mbm_events_attr;
	else
		intel_cqm_events_group.attrs = intel_mbm_events_attr;

	for_each_possible_cpu(i) {
		mbm_socket_max = max(mbm_socket_max,
				     topology_physical_package_id(i));
	}
	mbm_socket_max++;

	array_size = (cqm_max_rmid + 1) * mbm_socket_max;
	mbm_local = kzalloc_node(sizeof(struct sample) * array_size,
				 GFP_KERNEL, NUMA_NO_NODE);
	if (!mbm_local) {
		ret = -ENOMEM;
		goto free_str;
	}

	mbm_total = kzalloc_node(sizeof(struct sample) * array_size,
				 GFP_KERNEL, NUMA_NO_NODE);
	if (!mbm_total) {
		ret = -ENOMEM;
		goto free_local;
	}
	event_attr_intel_cqm_local_bw_scale.event_str = str;
	event_attr_intel_cqm_total_bw_scale.event_str = str;
	event_attr_intel_cqm_avg_local_bw_scale.event_str = str;
	event_attr_intel_cqm_avg_total_bw_scale.event_str = str;
	return 0;
free_local:
	kfree(mbm_local);
free_str:
	kfree(str);
	is_mbm = false;
	return ret;
}

static int __init intel_cqm_init(void)
{
	char *str = NULL, scale[20];
	int i, cpu, ret = 0;

	if ((!x86_match_cpu(intel_cqm_match)) &&
	    (!x86_match_cpu(intel_mbm_match)))
		return -ENODEV;

	cqm_l3_scale = boot_cpu_data.x86_cache_occ_scale;

	/*
	 * It's possible that not all resources support the same number
	 * of RMIDs. Instead of making scheduling much more complicated
	 * (where we have to match a task's RMID to a cpu that supports
	 * that many RMIDs) just find the minimum RMIDs supported across
	 * all cpus.
	 *
	 * Also, check that the scales match on all cpus.
	 */
	cpu_notifier_register_begin();

	for_each_online_cpu(cpu) {
		struct cpuinfo_x86 *c = &cpu_data(cpu);

		if (c->x86_cache_max_rmid < cqm_max_rmid)
			cqm_max_rmid = c->x86_cache_max_rmid;

		if (c->x86_cache_occ_scale != cqm_l3_scale) {
			pr_err("Multiple LLC scale values, disabling\n");
			ret = -EINVAL;
			goto out;
		}
	}
	if (x86_match_cpu(intel_cqm_match)) {
		cqm_llc_occ = true;
		intel_cqm_events_group.attrs = intel_cqm_events_attr;

	/*
	 * A reasonable upper limit on the max threshold is the number
	 * of lines tagged per RMID if all RMIDs have the same number of
	 * lines tagged in the LLC.
	 *
	 * For a 35MB LLC and 56 RMIDs, this is ~1.8% of the LLC.
	 */
	__intel_cqm_max_threshold =
		boot_cpu_data.x86_cache_size * 1024 / (cqm_max_rmid + 1);

	snprintf(scale, sizeof(scale), "%u", cqm_l3_scale);
	str = kstrdup(scale, GFP_KERNEL);
	if (!str) {
		ret = -ENOMEM;
		goto out;
	}

	event_attr_intel_cqm_llc_scale.event_str = str;
	}
	ret = intel_mbm_init();
	if ((ret) && (!cqm_llc_occ))
		goto out;
	ret = intel_cqm_setup_rmid_cache();
	if (ret)
		goto out;

	for_each_online_cpu(i) {
		ret = intel_cqm_cpu_starting(i);
		if (ret)
			goto out;
		cqm_pick_event_reader(i);
	}

	__perf_cpu_notifier(intel_cqm_cpu_notifier);

	ret = perf_pmu_register(&intel_cqm_pmu, "intel_cqm", -1);
	if (ret)
		pr_err("Intel CQM perf registration failed: %d\n", ret);
	else
		pr_info("Intel CQM monitoring enabled\n");

out:
	cpu_notifier_register_done();

	if (ret) {
		kfree(str);
		if (is_mbm) {
			kfree(mbm_local);
			kfree(mbm_total);
		}
	}
	return ret;
}
device_initcall(intel_cqm_init);
