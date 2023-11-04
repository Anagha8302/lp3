import random

def randomised_quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return randomised_quick_sort(left) + middle + randomised_quick_sort(right)

if __name__ == "__main__":
    
    user_input = input("Enter the numbers in list sepersted by spaces: ")
    arr = [int(x) for x in user_input.split()]
    
    sorted_randomised = randomised_quick_sort(arr.copy())
    
    print("Original List: ", arr)
    print("Sorted List: ", sorted_randomised)