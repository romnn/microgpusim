use serde_yaml::Value;

#[derive(Debug, Clone, PartialOrd, PartialEq, Hash)]
pub(crate) enum PrimitiveValue {
    String(String),
    Number(serde_yaml::Number),
    Bool(bool),
}

impl std::fmt::Display for PrimitiveValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::String(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Number(v) => write!(f, "{v}"),
        }
    }
}

#[derive(PartialEq, Eq)]
struct OrderedValue(Value);

impl std::cmp::PartialOrd for OrderedValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (&self.0, &other.0) {
            (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            (Value::Number(a), Value::Number(b)) => a.as_f64().partial_cmp(&b.as_f64()),
            (Value::Null, Value::Null) => Some(std::cmp::Ordering::Equal),
            _ => None,
        }
    }
}

pub(crate) fn flatten(value: serde_yaml::Value, sort: bool) -> Vec<PrimitiveValue> {
    use try_partialord::TrySort;
    match value {
        Value::Bool(v) => vec![PrimitiveValue::Bool(v)],
        Value::Number(v) => vec![PrimitiveValue::Number(v)],
        Value::String(v) => vec![PrimitiveValue::String(v)],
        Value::Sequence(v) => v.into_iter().flat_map(|s| flatten(s, sort)).collect(),
        Value::Mapping(v) => {
            let mut out = Vec::new();
            let mut items: Vec<_> = v.clone().into_iter().collect();
            if sort && items
                    .try_sort_by_key(|(key, _value)| Some(OrderedValue(key.clone())))
                    .is_err() {
                // undo
                items = v.into_iter().collect();
            }
            for (k, vv) in items {
                out.extend(flatten(k, sort));
                out.extend(flatten(vv, sort));
            }
            out
        }
        // skip
        Value::Null | serde_yaml::Value::Tagged(_) => vec![],
    }
}
