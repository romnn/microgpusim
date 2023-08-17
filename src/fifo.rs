use std::collections::VecDeque;

pub trait Queue<T>:
    std::iter::IntoIterator<Item = T> + Send + Sync + std::fmt::Display + 'static
{
    fn new<S: ToString>(name: S, min_size: Option<usize>, max_size: Option<usize>) -> Self;
    fn enqueue(&mut self, value: T);
    fn dequeue(&mut self) -> Option<T>;
    fn first(&self) -> Option<&T>;
    fn full(&self) -> bool;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn can_fit(&self, n: usize) -> bool;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fifo<T> {
    inner: VecDeque<T>,
    min_size: Option<usize>,
    max_size: Option<usize>,
}

impl<T> std::iter::IntoIterator for Fifo<T> {
    type Item = T;
    type IntoIter = std::collections::vec_deque::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<T> std::fmt::Display for Fifo<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Fifo({:>2}/{:<2}){:#?}",
            self.inner.len(),
            self.max_size
                .map(|max| max.to_string())
                .as_deref()
                .unwrap_or(""),
            self.inner
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
        )
    }
}

impl<T> Fifo<T> {
    #[must_use]
    pub fn iter(&self) -> std::collections::vec_deque::Iter<T> {
        self.inner.iter()
    }
}

impl<T> Queue<T> for Fifo<T>
where
    T: Send + Sync + std::fmt::Display + 'static,
{
    fn new<S: ToString>(_name: S, min_size: Option<usize>, max_size: Option<usize>) -> Self {
        Self {
            inner: VecDeque::new(),
            min_size,
            max_size,
        }
    }

    fn enqueue(&mut self, value: T) {
        self.inner.push_back(value);
    }

    fn dequeue(&mut self) -> Option<T> {
        self.inner.pop_front()
    }

    fn first(&self) -> Option<&T> {
        self.inner.get(0)
    }

    fn full(&self) -> bool {
        // log::trace!(
        //     "FIFO full? max len={:?} length={}",
        //     self.max_size,
        //     self.inner.len()
        // );
        match self.max_size {
            Some(max) => self.inner.len() >= max,
            None => false,
        }
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn can_fit(&self, n: usize) -> bool {
        // m_max_len && m_length + size - 1 >= m_max_len
        match self.max_size {
            // Some(max) => self.inner.len() + n - 1 < max,
            Some(max) => self.inner.len() + n <= max,
            None => true,
        }
    }
}
