/*
 * @author Hisashi Ikari
 */

#ifndef TINYCV_BASE_MEMORY_H
#define TINYCV_BASE_MEMORY_H

#include <memory>

namespace tinycv {
namespace memory {

template<long S, typename T, class A> struct ptr_buffer {
    inline void push(T* element) { _elements[_size++] = element; }
    inline T* pop() { return _elements[--_size]; }
    inline bool full() { return _size == S; }
    inline bool empty() { return _size == 0; }
    inline A& allocator() { return _allocator; }    
    private:
        A _allocator;    
        T* _elements[S];
        long _size;
};

template<typename T, typename A, typename B = ptr_buffer<300000, T, A>> B* deallocate() {
    thread_local std::unique_ptr<B, void(*)(B*)> buffer_ptr(new B(), [](B* buffer) {
        while (!buffer->empty()) buffer->allocator().deallocate(buffer->pop(), 1);
        delete buffer;
    });
    return buffer_ptr.get();
}

template<typename T, typename A, typename... S> T* allocate(S&&...args) {
    auto* buffer = deallocate<T, A>();
    T* result = (buffer->empty()) ? buffer->allocator().allocate(1) : buffer->pop();
    buffer->allocator().construct(result, std::forward<S>(args)...);
    return result;
}

template<typename T, typename P, typename A, typename... S> P make_ptr(S&&...args) {
    return P(allocate<T, A>(std::forward<S>(args)...), [](T* memory_ptr) {
        auto* buffer = deallocate<T, A>();
        buffer->allocator().destroy(memory_ptr);
        if (buffer->empty()) buffer->allocator().deallocate(memory_ptr, 1);
        else buffer->push(memory_ptr);
    });
}

}; // end of namespace

template<typename T, typename... S> std::shared_ptr<T> make_shared(S&&...args) {
    return memory::make_ptr<T, std::shared_ptr<T>, std::allocator<T>>(std::forward<S>(args)...);
}

}; // end of namespace

#endif

