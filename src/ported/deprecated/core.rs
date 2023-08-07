// pub fn occupy_resource_for_block(&mut self, kernel: &KernelInfo, _occupy: bool) -> bool {
//     let thread_block_size = self.inner.config.threads_per_block_padded(kernel);
//     if self.inner.num_occupied_threads + thread_block_size
//         > self.inner.config.max_threads_per_core
//     {
//         return false;
//     }
//     if self
//         .find_available_hw_thread_id(thread_block_size, false)
//         .is_none()
//     {
//         return false;
//     }
//     unimplemented!("occupy resource for block");
// }
