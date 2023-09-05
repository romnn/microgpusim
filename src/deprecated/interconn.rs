// impl Packet {
//     pub fn new(
//         #[cfg(feature = "stats")]
//         {
//             let mut stats = self.stats.lock();
//             let access_kind = *fetch.access_kind();
//             debug_assert_eq!(fetch.is_write(), access_kind.is_write());
//             stats.accesses.inc(access_kind, 1);
//         }
//
//         let dest_sub_partition_id = fetch.sub_partition_id();
//         let mem_dest = self.config.mem_id_to_device_id(dest_sub_partition_id);
//
//         log::debug!(
//             "cluster {} icnt_inject_request_packet({}) dest sub partition id={} dest mem node={}",
//             self.cluster_id,
//             fetch,
//             dest_sub_partition_id,
//             mem_dest
//         );
//
//         // The packet size varies depending on the type of request:
//         // - For write request and atomic request, packet contains the data
//         // - For read request (i.e. not write nor atomic), packet only has control metadata
//         let packet_size = if !fetch.is_write() && !fetch.is_atomic() {
//             fetch.control_size()
//         } else {
//             // todo: is that correct now?
//             fetch.size()
//             // fetch.data_size
//         };
//         // m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
//         fetch.status = mem_fetch::Status::IN_ICNT_TO_MEM;
//
//         // if let Packet::Fetch(fetch) = packet {
//         fetch.pushed_cycle = Some(time);
//
//         // self.interconn_queue
//         //     .push_back((mem_dest, fetch, packet_size));
//         // self.interconn
//         //     .push(self.cluster_id, mem_dest, Packet::Fetch(fetch), packet_size);
//         self.interconn_port
//             .lock()
//             .push_back((mem_dest, fetch, packet_size));
// }

// TODO: use a direct connection interface here
// THIS is just plain wrong and not thread safe
// impl<Q> MemFetchInterface for L2Interface<Q>
// where
//     Q: super::fifo::Queue<mem_fetch::MemFetch>,
// {
//     fn full(&self, _size: u32, _write: bool) -> bool {
//         self.l2_to_dram_queue.lock().full()
//     }
//
//     fn push(&self, mut fetch: mem_fetch::MemFetch, _time: u64) {
//         fetch.set_status(mem_fetch::Status::IN_PARTITION_L2_TO_DRAM_QUEUE, 0);
//         log::debug!("l2 interface push l2_to_dram_queue");
//         self.l2_to_dram_queue.lock().enqueue(fetch);
//     }
// }
//
// impl<T> Into<T> for Packet<T> {
//     fn into(self) -> T {
//         self.data
//     }
// }

// impl<T> From<Packet<T>> for T {
//     // fn from(self) -> T {
//     //     self.data
//     // }
// }
//

#[derive()]
pub struct CoreMemoryInterface<P> {
    pub config: Arc<config::GPU>,
    pub stats: Arc<Mutex<stats::Stats>>,
    pub cluster_id: usize,
    pub interconn: Arc<dyn Interconnect<P>>,
    pub interconn_port: Port,
}

impl<P> std::fmt::Debug for CoreMemoryInterface<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("CoreMemoryInterface").finish()
    }
}

impl MemFetchInterface for CoreMemoryInterface<Packet> {
    fn full(&self, size: u32, write: bool) -> bool {
        let request_size = if write {
            size
        } else {
            u32::from(mem_fetch::READ_PACKET_SIZE)
        };
        !self.interconn.has_buffer(self.cluster_id, request_size)
    }

    fn push(&self, mut fetch: mem_fetch::MemFetch, time: u64) {
        // self.core.interconn_simt_to_mem(fetch.get_num_flits(true));
        // self.cluster.interconn_inject_request_packet(fetch);

        {
            let mut stats = self.stats.lock();
            let access_kind = *fetch.access_kind();
            debug_assert_eq!(fetch.is_write(), access_kind.is_write());
            stats.accesses.inc(access_kind, 1);
        }

        let dest_sub_partition_id = fetch.sub_partition_id();
        let mem_dest = self.config.mem_id_to_device_id(dest_sub_partition_id);

        log::debug!(
            "cluster {} icnt_inject_request_packet({}) dest sub partition id={} dest mem node={}",
            self.cluster_id,
            fetch,
            dest_sub_partition_id,
            mem_dest
        );

        // The packet size varies depending on the type of request:
        // - For write request and atomic request, packet contains the data
        // - For read request (i.e. not write nor atomic), packet only has control metadata
        let packet_size = if !fetch.is_write() && !fetch.is_atomic() {
            fetch.control_size()
        } else {
            // todo: is that correct now?
            fetch.size()
            // fetch.data_size
        };
        // m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
        fetch.status = mem_fetch::Status::IN_ICNT_TO_MEM;

        // if let Packet::Fetch(fetch) = packet {
        fetch.pushed_cycle = Some(time);

        // self.interconn_queue
        //     .push_back((mem_dest, fetch, packet_size));
        // self.interconn
        //     .push(self.cluster_id, mem_dest, Packet::Fetch(fetch), packet_size);
        self.interconn_port
            .lock()
            .push_back((mem_dest, fetch, packet_size));
    }
}

// fn push_all(&self, _src: usize, packets: Vec<(usize, P, u32)>) {
//     assert!(self.has_buffer(src_device, size));
//
//     let is_memory_node = self.num_subnets > 1 && dest_device >= self.num_cores;
//     let subnet = usize::from(is_memory_node);
//     // log::debug!(
//     //     "{}: {size} bytes from device {src_device} to {dest_device} (subnet {subnet})",
//     //     style(format!("INTERCONN PUSH {packet}")).bold(),
//     // );
//
//     let mut queue = self.output_queue[subnet][dest_device][0]lock();
//     queue.extend(packets);
// }

// fn iter(&self) -> Box<dyn Iterator<Item = (uVecDeque<P>>>{
//     // let mut queue = self.output_queue[subnet][dest_device][0]lock();
//     // for (subnet_idx, subnet) in &self.output_queue.enumerate() {
//     // for (subnet_idx, subnet) in &self.output_queue.enumerate() {
//     //     for (nodex_idx, node) in subnet.iter().enumerate() {
//     //         for (class_idx, class) in node.iter().enumerate() {
//     //             // let _test = classlock();
//     //             // self.output_queue[subnet][node][class];
//     //         }
//     //     }
//     // }
// }

// fn sort(&self, order: Option<Vec>) {
//     // let mut queue = self.output_queue[subnet][dest_device][0]lock();
//     for subnet in &self.output_queue {
//         for node in subnet {
//             for class in node {
//                 let _test = classlock();
//                 // self.output_queue[subnet][node][class];
//             }
//         }
//     }
// }

// fn commit(&self) {
//     self.commit_queue
// }
//
// fn commit(&self);

// fn push_all(&self, _src: usize, packets: Vec<(usize, P, u32)>);
// fn iter(&self) -> Box<dyn Iterator<Item = (VecDeque<P>>>;
// fn sort(&self);
