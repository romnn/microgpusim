pub use std::sync::atomic;
pub use std::sync::Arc;

#[cfg(feature = "parking_lot")]
pub mod parking_lot {

    /// A mutex
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized>(pub parking_lot::Mutex<T>);

    impl<T> Mutex<T> {
        #[must_use]
        // #[inline]
        pub fn new(value: T) -> Self {
            Self(parking_lot::Mutex::new(value))
        }

        #[must_use]
        // #[inline]
        pub fn into_inner(self) -> T {
            self.0.into_inner()
        }
    }

    impl<T: ?Sized> Mutex<T> {
        // #[inline]
        pub fn lock(&self) -> parking_lot::MutexGuard<T> {
            self.0.lock()
        }

        // #[inline]
        pub fn try_lock(&self) -> parking_lot::MutexGuard<T> {
            // self.0.lock()
            self.0.try_lock().unwrap()
        }
    }

    /// A fair mutex
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct FairMutex<T: ?Sized>(pub parking_lot::FairMutex<T>);

    impl<T> FairMutex<T> {
        #[must_use]
        // #[inline]
        pub fn new(value: T) -> Self {
            Self(parking_lot::FairMutex::new(value))
        }

        #[must_use]
        // #[inline]
        pub fn into_inner(self) -> T {
            self.0.into_inner()
        }
    }

    impl<T: ?Sized> FairMutex<T> {
        // #[inline]
        pub fn lock(&self) -> parking_lot::FairMutexGuard<T> {
            self.0.lock()
        }

        // #[inline]
        pub fn try_lock(&self) -> parking_lot::FairMutexGuard<T> {
            // self.0.lock()
            self.0.try_lock().unwrap()
        }
    }

    /// A read-write lock
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct RwLock<T: ?Sized>(pub parking_lot::RwLock<T>);

    impl<T> RwLock<T> {
        #[must_use]
        // #[inline]
        pub fn new(value: T) -> RwLock<T> {
            Self(parking_lot::RwLock::new(value))
        }

        #[must_use]
        // #[inline]
        pub fn into_inner(self) -> T {
            self.0.into_inner()
        }
    }

    impl<T: ?Sized> RwLock<T> {
        // #[inline]
        pub fn read(&self) -> parking_lot::RwLockReadGuard<T> {
            self.0.read()
        }

        // #[inline]
        pub fn try_read(&self) -> parking_lot::RwLockReadGuard<T> {
            // self.0.read()
            self.0.try_read().unwrap()
        }

        // #[inline]
        pub fn write(&self) -> parking_lot::RwLockWriteGuard<T> {
            self.0.write()
        }

        // #[inline]
        pub fn try_write(&self) -> parking_lot::RwLockWriteGuard<T> {
            // self.0.write()
            self.0.try_write().unwrap()
        }
    }
}

#[cfg(not(feature = "parking_lot"))]
pub mod default {
    /// A mutex
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized>(pub std::sync::Mutex<T>);

    impl<T> Mutex<T> {
        #[must_use]
        // #[inline]
        pub fn new(value: T) -> Self {
            Self(std::sync::Mutex::new(value))
        }

        #[must_use]
        // #[inline]
        pub fn into_inner(self) -> T {
            self.0.into_inner().unwrap()
        }
    }

    impl<T: ?Sized> Mutex<T> {
        // #[inline]
        pub fn lock(&self) -> std::sync::MutexGuard<T> {
            self.0.lock().unwrap()
        }

        // #[inline]
        pub fn try_lock(&self) -> std::sync::MutexGuard<T> {
            self.0.try_lock().unwrap()
            // self.0.lock().unwrap()
        }
    }

    /// A read-write lock
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct RwLock<T: ?Sized>(pub std::sync::RwLock<T>);

    impl<T> RwLock<T> {
        #[must_use]
        // #[inline]
        pub fn new(value: T) -> RwLock<T> {
            Self(std::sync::RwLock::new(value))
        }

        #[must_use]
        // #[inline]
        pub fn into_inner(self) -> T {
            self.0.into_inner().unwrap()
        }
    }

    impl<T: ?Sized> RwLock<T> {
        // #[inline]
        pub fn read(&self) -> std::sync::RwLockReadGuard<T> {
            self.0.read().unwrap()
        }

        // #[inline]
        pub fn try_read(&self) -> std::sync::RwLockReadGuard<T> {
            self.0.try_read().unwrap()
            // self.0.read().unwrap()
        }

        // #[inline]
        pub fn write(&self) -> std::sync::RwLockWriteGuard<T> {
            self.0.write().unwrap()
        }

        // #[inline]
        pub fn try_write(&self) -> std::sync::RwLockWriteGuard<T> {
            self.0.try_write().unwrap()
            // self.0.write().unwrap()
        }
    }
}

#[cfg(feature = "parking_lot")]
pub use self::parking_lot::{Mutex, RwLock};
// pub use self::parking_lot::{FairMutex as Mutex, RwLock};

#[cfg(not(feature = "parking_lot"))]
pub use default::{Mutex, RwLock};
