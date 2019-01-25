# I'm using a bottom-up non-recursive approach because I don't think I've
# written a non-recursive merge sort before. We push each element of the input
# onto a queue as a single-element list, and then traverse the queue in pairs
# and merge them, pushing them onto a different queue. Then we swap the queues
# and continue until there's only one element left, which is the result.
def mergesort(v):
    """
    Given a list of comparables, return a new list containing the same items in
    the input in sorted order. I say "comparables" but I only really test on
    integers, it's just that the code only uses the < operator. This
    implementation should be a stable sort but I don't have any tests for it.
    """

    if len(v) < 2:
        return list(v)

    # we could save space here by using a smarter queue/algorithm:
    # use a single queue, and in each iteration, make note of the first
    # element that we push onto it, and then stop the iteration when we reach
    # it again.
    front_queue = list()
    back_queue = list()
    for x in v:
        front_queue.append([x])

    while len(front_queue) > 1:
        for i in range(0, len(front_queue), 2):
            # if there are an odd number of items to merge, cheat and merge it
            # with the last element in the back_queue. if we don't, in cases with
            # eg 2^k+1 elements, at the last iteration  we'll end up with a
            # vector of size n-1 and a vector of size 1 in the queue
            #
            # i feel like there should be a better solution to this
            if i == len(front_queue) - 1:
                last = merge(front_queue[i], back_queue[-1])
                back_queue[-1] = last
            else:
                merged = merge(front_queue[i], front_queue[i + 1])
                back_queue.append(merged)
        tmp = front_queue
        front_queue = back_queue
        back_queue = tmp
        back_queue.clear()

    return front_queue.pop()


def merge(l, r):
    """ Given two sorted lists, merge them into a single, new sorted list """
    output = list()

    l_current = 0
    r_current = 0
    # While true because we break only when one of the lists are finished
    while True:
        # if either list is finished, append the rest of the other and return
        if len(l) == l_current:
            output.extend(r[r_current:])
            break
        if len(r) == r_current:
            output.extend(l[l_current:])
            break

        # otherwise, take the min and add it to the output
        l_item = l[l_current]
        r_item = r[r_current]
        if l_item < r_item:
            output.append(l_item)
            l_current += 1
        else:
            output.append(r_item)
            r_current += 1

    return output
