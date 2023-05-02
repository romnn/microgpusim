pub fn pad_to_multiple(n: usize, k: usize) -> usize {
    let rem = n % k;
    if rem != 0 {
        ((n / k) + 1) * k
    } else {
        n
    }
}
