#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/cluster.hpp");

        type cluster_bridge;

        #[must_use]
        fn get_core_sim_order(self: &cluster_bridge) -> UniquePtr<CxxVector<u32>>;
    }

    // explicit instantiation for cluster_bridge to implement VecElement
    impl CxxVector<cluster_bridge> {}
}

pub use ffi::*;
