def recur(n):
    if n <= 1:
        return n
    else:
        return(recur(n-1) + recur(n-2))
    
def iterative(n):
    a = 0
    b = 1
    print(a,end=" ")
    print(b,end=" ")
    for i in range(2, n):
        print(a + b,end=" ")
        a, b = b, a + b

def find_fibonacci_step_count(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        count = 1
        while b < n:
            a, b = b, a + b
            count += 1
        return count

if __name__ == "__main__":
    n = int(input("Enter the nth number for series:"))
    step_count = find_fibonacci_step_count(n)
    if n<=0:
        print("Please enter the positive integer")
    else:
        print("Fibonacci series with recursion:")
        for i in range (n):
            print(recur(i),end=" ")
        print()
        print("Fibonacci series with iteration:")
        iterative(n)
        print()
        print(f"Step count to reach n = {n}: {step_count}")
