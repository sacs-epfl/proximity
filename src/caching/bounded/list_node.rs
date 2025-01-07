use std::sync::{Arc, Weak, Mutex};

pub(crate) type SharedNode<K, V> = Arc<Mutex<Node<K, V>>>;
pub(crate) type WeakSharedNode<K, V> = Weak<Mutex<Node<K, V>>>;

#[derive(Debug)]
pub struct Node<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
    pub(crate) prev: Option<WeakSharedNode<K, V>>,
    pub(crate) next: Option<SharedNode<K, V>>,
}

impl<K, V> Node<K, V> {
    pub fn new(key: K, value: V) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Node {
            key,
            value,
            prev: None,
            next: None,
        }))
    }
}
