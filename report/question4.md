# Assignment 3

<div align="right">
~ Submitted by Varun Jain
</div>

## Question 4

### For Logistic regression

Assuming:

- Number of features = m
- Number of samples = n
- Number of classes = k
- Number of Iterations = i

### For Learning

- Calculating loss takes `O[ndk]`.
- Forward Pass takes `O[ndk]`.
- `i` number of iterations

Thus total time complexity for Learning = `O[indk]`.

### For Predicting

- Forward Pass takes `O[ndk]`.

Thus Total time complexity = `O[ndk]`

### Space Complexity

Considering we are only storing weights and not counting Training data...

- Learning = `O[dk]`
- Predicting = `O[dk]`
