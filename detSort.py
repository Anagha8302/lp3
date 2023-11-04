def deterministic_quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return deterministic_quick_sort(left) + middle + deterministic_quick_sort(right)

if __name__ == "__main__":
    
    user_input = input("Enter the list of numbers seperated by spaces: ")
    arr = [int(x) for x in user_input.split()]
    
    sorted_deterministic = deterministic_quick_sort(arr.copy())
    
    print("Original List: ", arr)
    print("Sorted List: ", sorted_deterministic)