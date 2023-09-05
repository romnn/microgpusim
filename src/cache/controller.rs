/// Cache controller.
///
/// The cache controller intercepts read and write memory requests before passing them
/// on to the memory controller. It processes a request by dividing the address of the
/// request into three fields, the tag field, the set index field, and the data index field.
///
/// First, the controller uses the set index portion of the address to locate the cache
/// line within the cache memory that might hold the requested code or data. This cache
/// line contains the cache-tag and status bits, which the controller uses to determine
/// the actual data stored there.
///
/// The controller then checks the valid bit to determine if the cache line is active,
/// and compares the cache-tag to the tag field of the requested address.
/// If both the status check and comparison succeed, it is a cache hit.
/// If either the status check or comparison fails, it is a cache miss.
///
/// [ARM System Developer's Guide, 2004]
#[allow(clippy::module_name_repetitions)]
pub trait CacheController {}
