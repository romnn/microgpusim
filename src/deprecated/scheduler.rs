#[derive(Debug)]
pub struct LrrScheduler {
    inner: BaseSchedulerUnit,
}

impl SchedulerUnit for LrrScheduler {
    // impl<'a> SchedulerUnit for LrrScheduler<'a> {
    fn order_warps(
        &mut self,
        // out: &mut VecDeque<SchedulerWarp>,
        // warps: &mut Vec<SchedulerWarp>,
        // last_issued_warps: &Vec<SchedulerWarp>,
        // num_warps_to_add: usize,
    ) {
        self.inner.order_lrr();
        // let num_warps_to_add = self.inner.supervised_warps.len();
        // order_lrr(
        //     &mut self.inner.next_cycle_prioritized_warps,
        //     &mut self.inner.supervised_warps,
        //     &mut self.inner.last_supervised_issued_idx,
        //     // &mut self.inner.last_supervised_issued(),
        //     num_warps_to_add,
        // );
    }

    fn add_supervised_warp(&mut self, warp: CoreWarp) {
        self.inner.supervised_warps.push_back(warp);
        // self.inner.add_supervised_warp_id(warp_id);
    }

    fn prioritized_warps(&self) -> &VecDeque<CoreWarp> {
        self.inner.prioritized_warps()
    }

    // fn add_supervised_warp_id(&mut self, warp_id: usize) {
    //     self.inner.add_supervised_warp_id(warp_id);
    // }

    // fn done_adding_supervised_warps(&mut self) {
    //     self.inner.last_supervised_issued_idx = self.inner.supervised_warps.len();
    // }

    // fn cycle<I>(&mut self, core: &mut super::core::InnerSIMTCore<I>) {
    // fn cycle(&mut self, core: ()) {
    fn cycle(&mut self, issuer: &mut dyn super::core::WarpIssuer) {
        self.order_warps();
        self.inner.cycle(issuer);
    }
}

// impl<'a> LrrScheduler<'a> {
impl LrrScheduler {
    // fn order_warps(
    //     &self,
    //     out: &mut VecDeque<SchedulerWarp>,
    //     warps: &mut Vec<SchedulerWarp>,
    //     last_issued_warps: &Vec<SchedulerWarp>,
    //     num_warps_to_add: usize,
    // ) {
    //     todo!("scheduler unit: order warps")
    // }

    // pub fn new(
    //     id: usize,
    //     // warps: &'a Vec<SchedulerWarp>,
    //     warps: Vec<CoreWarp>,
    //     // warps: &'a Vec<Option<SchedulerWarp>>,
    //     // mem_out: &'a register_set::RegisterSet,
    //     // core: &'a super::core::InnerSIMTCore,
    //     scoreboard: Arc<RwLock<scoreboard::Scoreboard>>,
    //     stats: Arc<Mutex<stats::Stats>>,
    //     config: Arc<GPUConfig>,
    // ) -> Self {
    //     // todo!("lrr scheduler: new");
    //     let inner = BaseSchedulerUnit::new(
    //         id, // mem_out, core,
    //         warps, scoreboard, stats, config,
    //     );
    //     Self { inner }
    // }

    // lrr_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
    //               Scoreboard *scoreboard, simt_stack **simt,
    //               std::vector<shd_warp_t *> *warp, register_set *sp_out,
    //               register_set *dp_out, register_set *sfu_out,
    //               register_set *int_out, register_set *tensor_core_out,
    //               std::vector<register_set *> &spec_cores_out,
    //               register_set *mem_out, int id)
    //     : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
    //                      sfu_out, int_out, tensor_core_out, spec_cores_out,
    //                      mem_out, id) {}

    // virtual void order_warps();
}

fn order_rrr(
    &mut self,
    // out: &mut VecDeque<SchedulerWarp>,
    // warps: &mut Vec<SchedulerWarp>,
    // std::vector<T> &result_list, const typename std::vector<T> &input_list,
    // const typename std::vector<T>::const_iterator &last_issued_from_input,
    // unsigned num_warps_to_add)
) {
    unimplemented!("order rrr is untested");
    let num_warps_to_add = self.supervised_warps.len();
    let out = &mut self.next_cycle_prioritized_warps;
    // order_lrr(
    //     &mut self.inner.next_cycle_prioritized_warps,
    //     &mut self.inner.supervised_warps,
    //     &mut self.inner.last_supervised_issued_idx,
    //     // &mut self.inner.last_supervised_issued(),
    //     num_warps_to_add,
    // );

    out.clear();

    let current_turn_warp_ref = self.warps.get(self.current_turn_warp).unwrap();
    let current_turn_warp = current_turn_warp_ref.try_borrow().unwrap();
    // .as_ref()
    // .unwrap();

    if self.num_issued_last_cycle > 0
        || current_turn_warp.done_exit()
        || current_turn_warp.waiting()
    {
        // std::vector<shd_warp_t *>::const_iterator iter =
        //   (last_issued_from_input == input_list.end()) ?
        //     input_list.begin() : last_issued_from_input + 1;

        let mut iter = self
            .supervised_warps
            .iter()
            .skip(self.last_supervised_issued_idx + 1)
            .chain(self.supervised_warps.iter());

        for w in iter.take(num_warps_to_add) {
            let warp = w.try_borrow().unwrap();
            let warp_id = warp.warp_id;
            if !warp.done_exit() && !warp.waiting() {
                out.push_back(w.clone());
                self.current_turn_warp = warp_id;
                break;
            }
        }
        // for (unsigned count = 0; count < num_warps_to_add; ++iter, ++count) {
        //   if (iter == input_list.end()) {
        //   iter = input_list.begin();
        //   }
        //   unsigned warp_id = (*iter)->get_warp_id();
        //   if (!(*iter)->done_exit() && !(*iter)->waiting()) {
        //     result_list.push_back(*iter);
        //     m_current_turn_warp = warp_id;
        //     break;
        //   }
        // }
    } else {
        out.push_back(current_turn_warp_ref.clone());
    }
}

fn order_lrr(
    &mut self,
    // out: &mut VecDeque<SchedulerWarp>,
    // warps: &mut Vec<SchedulerWarp>,
    // // last_issued_warps: &Vec<SchedulerWarp>,
    // // last_issued_warps: impl Iterator<Item=SchedulerWarp>,
    // // last_issued_warps: &mut std::slice::Iter<'_, SchedulerWarp>,
    // // last_issued_warps: impl Iterator<Item = &'a SchedulerWarp>,
    // last_issued_warp_idx: &mut usize,
    // num_warps_to_add: usize,
) {
    unimplemented!("order lrr is not tested");
    let num_warps_to_add = self.supervised_warps.len();
    let out = &mut self.next_cycle_prioritized_warps;

    debug_assert!(num_warps_to_add <= self.warps.len());
    out.clear();
    // if last_issued_warps
    //   typename std::vector<T>::const_iterator iter = (last_issued_from_input == input_list.end()) ? input_list.begin()
    //                                                    : last_issued_from_input + 1;
    //
    let mut last_issued_iter = self.warps.iter().skip(self.last_supervised_issued_idx);

    let mut iter = last_issued_iter.chain(self.warps.iter());
    // .filter_map(|x| x.as_ref());
    // .filter_map(|x| x.as_ref());

    out.extend(iter.take(num_warps_to_add).cloned());
    // for count in 0..num_warps_to_add {
    //     let Some(warp) = iter.next() else {
    //         return;
    //     };
    //     // if (iter == input_list.end()) {
    //     //   iter = input_list.begin();
    //     // }
    //     out.push_back(warp.clone());
    // }
    // todo!("order lrr: order warps")
}
