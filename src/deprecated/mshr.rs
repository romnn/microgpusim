/// check `is_read_after_write_pending`
// #[allow(dead_code)]
// pub fn is_read_after_write_pending(&self, block_addr: address) -> bool {
//     let mut write_found = false;
//     for fetch in &self.entries[&block_addr].list {
//         if fetch.is_write() {
//             // pending write
//             write_found = true;
//         } else if write_found {
//             // pending read and previous write
//             return true;
//         }
//     }
//     return false;
// }

