use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

pub(crate) type SharedNode<K, V> = Rc<RefCell<Node<K, V>>>;
pub(crate) type WeakSharedNode<K, V> = Weak<RefCell<Node<K, V>>>;

#[derive(Debug)]
pub struct Node<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
    pub(crate) prev: Option<WeakSharedNode<K, V>>,
    pub(crate) next: Option<SharedNode<K, V>>,
}

impl<K, V> Node<K, V> {
    pub fn new(key: K, value: V) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Node {
            key,
            value,
            prev: None,
            next: None,
        }))
    }
}
