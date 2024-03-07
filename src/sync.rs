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
        pub fn new(value: T) -> Self {
            Self(parking_lot::Mutex::new(value))
        }

        #[must_use]
        pub fn into_inner(self) -> T {
            self.0.into_inner()
        }
    }

    impl<T: ?Sized> Mutex<T> {
        pub fn lock(&self) -> parking_lot::MutexGuard<T> {
            self.0.lock()
        }

        pub fn try_lock(&self) -> parking_lot::MutexGuard<T> {
            match self.0.try_lock() {
                Some(guard) => guard,
                None => {
                    println!("{}", std::backtrace::Backtrace::force_capture());
                    panic!("try_lock() would block");
                }
            }
        }
    }

    /// A fair mutex
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct FairMutex<T: ?Sized>(pub parking_lot::FairMutex<T>);

    impl<T> FairMutex<T> {
        #[must_use]
        pub fn new(value: T) -> Self {
            Self(parking_lot::FairMutex::new(value))
        }

        #[must_use]
        pub fn into_inner(self) -> T {
            self.0.into_inner()
        }
    }

    impl<T: ?Sized> FairMutex<T> {
        pub fn lock(&self) -> parking_lot::FairMutexGuard<T> {
            self.0.lock()
        }

        pub fn try_lock(&self) -> parking_lot::FairMutexGuard<T> {
            match self.0.try_lock() {
                Some(guard) => guard,
                None => {
                    println!("{}", std::backtrace::Backtrace::force_capture());
                    panic!("try_lock() would block");
                }
            }
        }
    }

    /// A read-write lock
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct RwLock<T: ?Sized>(pub parking_lot::RwLock<T>);

    impl<T> RwLock<T> {
        #[must_use]
        pub fn new(value: T) -> RwLock<T> {
            Self(parking_lot::RwLock::new(value))
        }

        #[must_use]
        pub fn into_inner(self) -> T {
            self.0.into_inner()
        }
    }

    impl<T: ?Sized> RwLock<T> {
        pub fn read(&self) -> parking_lot::RwLockReadGuard<T> {
            self.0.read()
        }

        pub fn try_read(&self) -> parking_lot::RwLockReadGuard<T> {
            self.0.try_read().unwrap()
        }

        pub fn write(&self) -> parking_lot::RwLockWriteGuard<T> {
            self.0.write()
        }

        pub fn try_write(&self) -> parking_lot::RwLockWriteGuard<T> {
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
        pub fn new(value: T) -> Self {
            Self(std::sync::Mutex::new(value))
        }

        #[must_use]
        pub fn into_inner(self) -> T {
            self.0.into_inner().unwrap()
        }
    }

    impl<T: ?Sized> Mutex<T> {
        pub fn lock(&self) -> std::sync::MutexGuard<T> {
            self.0.lock().unwrap()
        }

        pub fn try_lock(&self) -> std::sync::MutexGuard<T> {
            match self.0.try_lock() {
                Err(err) => {
                    println!("{}: {}", err, std::backtrace::Backtrace::force_capture());
                    panic!("{}", err);
                }
                Ok(guard) => guard,
            }
        }
    }

    /// A read-write lock
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct RwLock<T: ?Sized>(pub std::sync::RwLock<T>);

    impl<T> RwLock<T> {
        #[must_use]
        pub fn new(value: T) -> RwLock<T> {
            Self(std::sync::RwLock::new(value))
        }

        #[must_use]
        pub fn into_inner(self) -> T {
            self.0.into_inner().unwrap()
        }
    }

    impl<T: ?Sized> RwLock<T> {
        pub fn read(&self) -> std::sync::RwLockReadGuard<T> {
            self.0.read().unwrap()
        }

        pub fn try_read(&self) -> std::sync::RwLockReadGuard<T> {
            match self.0.try_read() {
                Err(err) => {
                    println!("{}: {}", err, std::backtrace::Backtrace::force_capture());
                    panic!("{}", err);
                }
                Ok(guard) => guard,
            }
        }

        pub fn write(&self) -> std::sync::RwLockWriteGuard<T> {
            self.0.write().unwrap()
        }

        pub fn try_write(&self) -> std::sync::RwLockWriteGuard<T> {
            match self.0.try_write() {
                Err(err) => {
                    println!("{}: {}", err, std::backtrace::Backtrace::force_capture());
                    panic!("{}", err);
                }
                Ok(guard) => guard,
            }
        }
    }
}

#[cfg(feature = "parking_lot")]
pub use self::parking_lot::{Mutex, RwLock};
#[cfg(feature = "parking_lot")]
pub type FairMutex<T> = self::parking_lot::FairMutex<T>;
// pub type FairMutex<T> = self::parking_lot::Mutex<T>;

#[cfg(not(feature = "parking_lot"))]
pub use default::{Mutex, RwLock};

#[cfg(not(feature = "parking_lot"))]
pub type FairMutex<T> = default::Mutex<T>;
