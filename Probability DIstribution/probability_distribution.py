import csv
import math


# -------------------------
# Load data
# -------------------------

def load_data(filename):

    heights = []

    with open(filename, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            # Parent + Child mixture
            heights.append(float(row["father"]))
            heights.append(float(row["childHeight"]))

    return heights



# -------------------------
# Gaussian probability
# -------------------------

def gaussian(x, mean, variance):

    return (
        1 / math.sqrt(2 * math.pi * variance)
    ) * math.exp(
        -((x - mean) ** 2) / (2 * variance)
    )



# -------------------------
# EM Algorithm
# -------------------------

def EM(data, iterations=10):


    # Better initialization

    average = sum(data) / len(data)

    mean1 = average - 5

    mean2 = average + 5


    variance1 = 25

    variance2 = 25


    pi1 = 0.5

    pi2 = 0.5



    print("\nTracking Table")

    print(
        "Iteration | mu1 | mu2 | var1 | var2 | pi1 | pi2 | LogLikelihood"
    )


    for iteration in range(iterations):


        responsibilities1 = []

        responsibilities2 = []



        # -----------------
        # E STEP
        # -----------------

        for x in data:


            prob1 = pi1 * gaussian(
                x,
                mean1,
                variance1
            )


            prob2 = pi2 * gaussian(
                x,
                mean2,
                variance2
            )


            total = prob1 + prob2


            r1 = prob1 / total

            r2 = prob2 / total


            responsibilities1.append(r1)

            responsibilities2.append(r2)




        # -----------------
        # M STEP
        # -----------------


        N1 = sum(responsibilities1)

        N2 = sum(responsibilities2)



        mean1 = sum(
            responsibilities1[i] * data[i]
            for i in range(len(data))
        ) / N1



        mean2 = sum(
            responsibilities2[i] * data[i]
            for i in range(len(data))
        ) / N2




        variance1 = sum(

            responsibilities1[i]
            *
            (data[i] - mean1) ** 2

            for i in range(len(data))

        ) / N1




        variance2 = sum(

            responsibilities2[i]
            *
            (data[i] - mean2) ** 2

            for i in range(len(data))

        ) / N2




        pi1 = N1 / len(data)

        pi2 = N2 / len(data)




        # -----------------
        # Log Likelihood
        # -----------------

        log_likelihood = 0


        for x in data:


            value = (

                pi1 * gaussian(
                    x,
                    mean1,
                    variance1
                )

                +

                pi2 * gaussian(
                    x,
                    mean2,
                    variance2
                )

            )


            log_likelihood += math.log(value)



        print(

            iteration,

            round(mean1,2),

            round(mean2,2),

            round(variance1,2),

            round(variance2,2),

            round(pi1,2),

            round(pi2,2),

            round(log_likelihood,2)

        )



    return (

        mean1,
        mean2,
        variance1,
        variance2,
        pi1,
        pi2

    )




# -------------------------
# Prediction
# -------------------------

def predict(height, model):


    mean1, mean2, var1, var2, pi1, pi2 = model



    # EM does not know labels
    # assign labels using means


    if mean1 < mean2:


        child_mean = mean1

        child_var = var1

        child_pi = pi1


        parent_mean = mean2

        parent_var = var2

        parent_pi = pi2


    else:


        child_mean = mean2

        child_var = var2

        child_pi = pi2


        parent_mean = mean1

        parent_var = var1

        parent_pi = pi1





    child_probability = (

        child_pi *
        gaussian(
            height,
            child_mean,
            child_var
        )

    )



    parent_probability = (

        parent_pi *
        gaussian(
            height,
            parent_mean,
            parent_var
        )

    )



    total = child_probability + parent_probability



    print("\nPrediction for:", height)


    print(
        "Child probability:",
        round(child_probability / total * 100,2),
        "%"
    )


    print(
        "Parent probability:",
        round(parent_probability / total * 100,2),
        "%"
    )





# -------------------------
# Run Program
# -------------------------


data = load_data("GaltonFamilies.csv")


model = EM(data)



test_height = float(
    input("\nEnter test height: ")
)


predict(test_height, model)