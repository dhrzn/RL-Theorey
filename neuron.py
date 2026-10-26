import numpy as np

#defining batches [current position given, target position]
first_pair = [2,8]
second_pair = [4, 1]
third_pair = [3, 7]
fourth_pair = [1, 9]
fifth_pair = [10, 0]

#defining weights and bias numbers [weights control the strength of how the neuron will respond to inputs (in this case the neuron will decide how hard it will respond to [x,t]) and its response would show in the activation function)
#Note: small weights and bias are chosen because if the weights are large, the weighted sum (z) gets large and tanh(z) saturates near 1 or -1
#when we update the weights with backpropagation (chain rule), the slope of tanh is 1 - tanh^2, which is about 0 when z is large,
#so the whole product of slopes is about 0 and the weights barely move (training gets stuck)
#near z = 0 that slope is about 1, so we start with small values to begin on the steep part
w1 = 0.1
w2 = 0.1
b = 0.1

pairs = [first_pair, second_pair, third_pair, fourth_pair, fifth_pair]
weighted_sum_results = []
tanh_results = []
loss_results = []
push_results = []

for x,t in pairs:
    #finding weighted sum for each pair (weighted sum is the combined strength of how the entire system would react based off of how hard it weighted the input values with w1 and w2)
    weighted_sum = w1 * x + w2 * t + b
    weighted_sum_results.append(weighted_sum)

    #label is the answer key: the correct push for this pair 
    label = np.tanh(t - x)
    tanh_results.append(label)

    #calculating tanh for the weighted sum [this is the activation function] (activation function is the final determination of how the system would react (-1 and 1 being the strongest pushes both left and right))
    push = np.tanh(weighted_sum)  #the neuron's own push
    push_results.append(push)
    #calculating updated position
    x_pos_updated = x + push

    #calculating loss: we compare the answer key (us) to what the neuron actually did, that's how we get the result of how it did
    #the loss is the squared gap between push and label (squared so the sign can't cancel anything out)
    loss = (push-label)**2
    loss_results.append(loss)

    #printing results
    print(f"Pair [{x}, {t}] --> Weighted sum: {weighted_sum:2f} --> label: {label:.6f} --> push: {push:.6f} --> x updated: {x_pos_updated:.2f} --> loss result: {loss}")

#calculating the MSE (mean squared error: square each gap, then take the mean)
# Note: (this basically describes how well our w1 and w2 guesses work for every case)
# Our goal is to find a w1 and w2 that works for every case and we try new ones till the MSE stops shrinking (once it stops shrinking thats how we know we have reached the best w1 and w2)
mean_loss_value = sum(loss_results) / len(loss_results)


#calculating backpropagation using [4,1] pair as an example
#the chain: weights --> weighted sum --> push -->  loss
#Note: the chain is this way because we compare the push to the answer key (label) [we compare the push because its dependent on weights we chose]
#Note: although we follow the same structure as LineEnv, LineEnv gets "graded" by: the push moves the cart, and we score where the cart ends up, however here in neuron.py, rather than considering the updated x position, we just compare the push straight to the label push (the push we got vs the push we should have gotten)
#note: slope = the thing that depends on what you change / the thing you change
x_sec, t_sec = pairs[1] #grabbing pairs from the list defined earlier
label_sec = tanh_results[1] #grabbing the answer key from the pair [4,1]
push_sec = push_results[1] #grabbing push results from list defined earlier
slope_1_w1 = x_sec #w1 is multiplied by x             [Note: slope 1 is the slope of weighted sum (how each of the weights affected weighted sum)]
slope_1_w2 = t_sec #w2 is multiplied by t
slope_1_b = 1 #b is multiplied by nothing 
slope_2 = 1 - (push_sec**2) #this slope belongs to the push, where we compare how the weighted sum affected push
slope_3 = 2*(push_sec-label_sec) #this slope belongs to the loss and is where we compare to answer key

#finding final slopes for each of the dials from slope1 (w1, w2, w3) so we can represent each slope individually since slope 1 doesnt contain just 1 slope
final_slope_w1 = slope_1_w1 * slope_2 * slope_3
final_slope_w2 = slope_1_w2 * slope_2 * slope_3
final_slope_b = slope_1_b * slope_2 * slope_3


print("\n--- Summary ---")
print("All initial values: ", weighted_sum_results)
print("All tanh values:  ", tanh_results)
print(f"Mean of losses:  {mean_loss_value:.6f}")
print(f"Slope1 : w1 = {slope_1_w1:.6f} | w2 = {slope_1_w2:.6f} | b = {slope_1_b:.6f} | Slope2 = {slope_2} | Slope3 = {slope_3}")
print(f"Final Slopes --> w1 = {final_slope_w1:.6f} | w2 = {final_slope_w2:.6f} | b = {final_slope_b:.6f}")

#note to self (whole thing):
#neuron.py is the brain that will play LineEnv later. it only ever sees ONE pair [x, t] and makes ONE push.
#weighted sum = w1*x + w2*t + b, activation = tanh of that, and the activation output is the push.
#the neuron never sees the answer key. the loss is calculated outside it, after the push already exists.
#label = the answer key (us). push = what the neuron actually did. loss = the squared gap between them.
#weights stay fixed for the whole batch. they only change AFTER the MSE, in the update step.
#now w1, w2, b are small (0.1). push is about +0.5 to +0.8 for every pair, so it still gets [4, 1] and [10, 0] wrong. the weights have to change before it can push left.
#next: slopes with the chain rule for w1, w2 and b, then the loop that updates them until the MSE stops shrinking.

