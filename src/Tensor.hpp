#ifndef CPPIR_TENSOR_HPP
#define CPPIR_TENSOR_HPP
#include <concepts>
#include <ranges>
#include <utility>
#include <cstdint>
#include <array>

namespace cppir {


struct Dyn_t {};
constexpr inline Dyn_t dyn_ext;

template<class T>
concept ITensorElement = std::semiregular<T> && std::is_object_v<T> && !std::is_const_v<T> && !std::is_volatile_v<T>;

template <class StorageT, class NumT, class IdxT>
concept ITensorStorage = std::movable<StorageT> && ITensorElement<NumT> && std::signed_integral<IdxT> && 
requires (StorageT& s, const StorageT& cs) 
{
    typename StorageT::value_type;
    requires std::same_as<typename StorageT::value_type, NumT>;

    typename StorageT::size_type;
    requires std::convertible_to<IdxT, typename StorageT::size_type>;

    requires std::ranges::contiguous_range<StorageT>;

    { s.data() } -> std::same_as<NumT*>;
    { cs.size() } -> std::same_as<IdxT>;
    { cs.data() } -> std::same_as<const NumT*>;
    { s[IdxT{}] } -> std::same_as<NumT&>;
    { cs[IdxT{}] } -> std::same_as<const NumT&>;
};

template <auto Extent>
using ExtentT = std::remove_cv_t<decltype(Extent)>;

template <auto Extent>
static constexpr bool is_dyn = std::same_as<ExtentT<Extent>, Dyn_t>;

template<std::signed_integral IndexT, auto Extent>
consteval bool valid_extent()
{
    using ExtentT = ExtentT<Extent>;

    if constexpr (std::same_as<ExtentT, Dyn_t>) {
        return true;
    } else if constexpr (std::same_as<ExtentT, IndexT>) {
        return std::in_range<IndexT>(Extent) && Extent > 0;
    } else {
        return false;
    }
}

template<uint8_t I, auto... Values>
struct NthValue;

template<uint8_t I, auto First, auto... Rest>
struct NthValue<I, First, Rest...> : NthValue<I - 1, Rest...> {};

template<auto First, auto... Rest>
struct NthValue<0, First, Rest...> { static constexpr auto value = First; };

template <std::signed_integral idx_t, auto... Extents>
class Extent
{
public:
    static constexpr uint8_t rank = sizeof...(Extents);
    static constexpr uint8_t dyn_rank = (uint8_t{0} + ... + uint8_t{is_dyn<Extents>});
    static constexpr uint8_t static_rank = rank - dyn_rank;

private:
    struct Empty_t {};
    using Storage = std::conditional_t<dyn_rank == 0, Empty_t, std::array<idx_t, dyn_rank>>;

    [[no_unique_address]] const Storage m_storage;

    template <uint8_t Axis>
    static constexpr auto extent = NthValue<Axis, Extents...>::value;

    template<auto Ext>
    static consteval idx_t static_value_or_zero()
    {
        if constexpr (is_dyn<Ext>) {
            return idx_t{0};
        } else {
            return static_cast<idx_t>(Ext);
        }
    }

    inline static constexpr std::array<bool, rank> dynamic_mask{ is_dyn<Extents>...  };

    inline static constexpr std::array<idx_t, rank> static_values{ static_value_or_zero<Extents>()...  };

public:
    constexpr Extent() requires (dyn_rank == 0) = default;

    template <std::convertible_to<idx_t>... Values>
        requires (dyn_rank > 0 && sizeof...(Values) == dyn_rank)
    constexpr explicit Extent(Values&&... values) : m_storage{static_cast<idx_t>(std::forward<Values>(values))...} {}

    template<uint8_t Axis> requires (Axis < rank)
    static consteval bool is_dynamic()
    {
        return is_dyn<extent<Axis>>;
    }

    template<uint8_t Axis> requires (Axis < rank)
    static consteval auto static_extent()
    {
        return extent<Axis>;
    }

    template<uint8_t Axis> requires (Axis < rank)
    constexpr idx_t get() const noexcept
    {
        if constexpr (is_dynamic<Axis>()) {
            return m_storage[dynamic_slot<Axis>];
        } else {
            return static_cast<IdxT>(specification<Axis>);
        }
    }


};

template <ITensorElement num_t, class storage_t, std::signed_integral idx_t, auto... Extents>
    requires (( valid_extent<idx_t, Extents>() && ... ) && ITensorStorage<storage_t, num_t, idx_t>)
class Tensor
{
public:
    using NumT = num_t;
    using StorageT = storage_t;
    using IdxT = idx_t;
    using size_type = IdxT;
    using value_type = NumT;



private:
    [[no_unique_address]] storage_t m_storage;

public:
    Tensor() = default;

    template <class... Args>
        requires std::constructible_from<StorageT, Args...>
    Tensor(Args&&... args) : m_storage(std::forward<Args>(args)...) {}

    template <class... Args>
        requires std::constructible_from<StorageT, Args...>
    Tensor(std::invocable<StorageT&> auto&& fn, Args&&... args) : m_storage(std::forward<Args>(args)...)
    {
        fn(m_storage);
    }

};

}  // namespace cppir

#endif