"""
TV Program Scheduling using a Genetic Algorithm - Streamlit interface
Built from the Week 4 tutorial "TV Scheduling - Genetic Algorithm.py".

Run locally:   streamlit run app.py
"""
import csv
import random

import pandas as pd
import streamlit as st

ORIGINAL_CSV = "program_ratings.csv"
MODIFIED_CSV = "program_ratings_modified.csv"


# ------------------------------------------------------------------
# Read the CSV file and convert it to {program: [ratings per hour]}
# (same as the tutorial)
# ------------------------------------------------------------------
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)  # skip the header
        for row in reader:
            if not row:
                continue
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
    return program_ratings


# ------------------------------------------------------------------
# GA functions (same operators as the tutorial)
# ------------------------------------------------------------------
def fitness_function(schedule, ratings):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating


def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2


def mutate(schedule, all_programs):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule


def initial_population(all_programs, n_slots, population_size):
    """Each schedule fills every time slot (06:00-23:00). Every program is
    used at least once, the remaining slots are filled randomly, then shuffled."""
    population = []
    for _ in range(population_size):
        schedule = all_programs.copy()
        while len(schedule) < n_slots:
            schedule.append(random.choice(all_programs))
        schedule = schedule[:n_slots]
        random.shuffle(schedule)
        population.append(schedule)
    return population


def genetic_algorithm(ratings, all_programs, n_slots, generations=100,
                      population_size=50, crossover_rate=0.8,
                      mutation_rate=0.2, elitism_size=2, seed=None):
    if seed is not None:
        random.seed(seed)

    population = initial_population(all_programs, n_slots, population_size)
    history = []

    for generation in range(generations):
        # Elitism
        population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
        history.append(fitness_function(population[0], ratings))
        new_population = [s.copy() for s in population[:elitism_size]]

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
    best = population[0]
    history.append(fitness_function(best, ratings))
    return best, history


# ------------------------------------------------------------------
# Streamlit interface
# ------------------------------------------------------------------
st.set_page_config(page_title="GA TV Scheduling", layout="wide")
st.title("📺 TV Program Scheduling using a Genetic Algorithm")

# ---- Data -------------------------------------------------------
st.sidebar.header("Dataset")
dataset = st.sidebar.radio("Program ratings to use",
                           ["Modified ratings", "Original ratings"])
csv_path = MODIFIED_CSV if dataset == "Modified ratings" else ORIGINAL_CSV
ratings = read_csv_to_dict(csv_path)

all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))
n_slots = len(all_time_slots)
slot_labels = [f"{h:02d}:00" for h in all_time_slots]

with st.expander("Program ratings used in this run"):
    st.dataframe(pd.DataFrame(ratings, index=slot_labels).T, use_container_width=True)

with st.expander("Changes made to the ratings (original → modified)"):
    original = read_csv_to_dict(ORIGINAL_CSV)
    modified = read_csv_to_dict(MODIFIED_CSV)
    changes = [
        {"Program": p, "Time Slot": slot_labels[i],
         "Original": original[p][i], "Modified": modified[p][i]}
        for p in original for i in range(n_slots)
        if original[p][i] != modified[p][i]
    ]
    st.dataframe(pd.DataFrame(changes), hide_index=True, use_container_width=True)

# ---- Fixed GA settings ------------------------------------------
st.sidebar.header("Other GA settings")
GEN = st.sidebar.number_input("Generations (GEN)", 10, 1000, 100, step=10)
POP = st.sidebar.number_input("Population size (POP)", 10, 500, 50, step=10)
EL_S = st.sidebar.number_input("Elitism size (EL_S)", 0, 10, 2)
SEED = st.sidebar.number_input("Random seed", 0, 9999, 42,
                               help="Same seed = repeatable results")

# ---- CO_R and MUT_R per trial -----------------------------------
st.subheader("Set the crossover rate and mutation rate for each trial")
st.caption("CO_R range: 0.00 – 0.95  |  MUT_R range: 0.01 – 0.05")

defaults = [(0.80, 0.02), (0.60, 0.05), (0.95, 0.01)]
cols = st.columns(3)
params = []
for i, col in enumerate(cols):
    with col:
        st.markdown(f"**Trial {i + 1}**")
        co_r = st.slider("Crossover Rate (CO_R)", 0.0, 0.95,
                         defaults[i][0], 0.01, key=f"co{i}")
        mut_r = st.slider("Mutation Rate (MUT_R)", 0.01, 0.05,
                          defaults[i][1], 0.01, key=f"mu{i}")
        params.append((co_r, mut_r))

# ---- Run --------------------------------------------------------
if st.button("▶ Run all 3 trials", type="primary"):
    summary, curves = [], {}
    result_cols = st.columns(3)
    for i, (co_r, mut_r) in enumerate(params):
        best, history = genetic_algorithm(
            ratings, all_programs, n_slots,
            generations=GEN, population_size=POP,
            crossover_rate=co_r, mutation_rate=mut_r,
            elitism_size=EL_S, seed=SEED,
        )
        total = fitness_function(best, ratings)
        table = pd.DataFrame({
            "Time Slot": slot_labels,
            "Program": best,
            "Rating": [ratings[p][s] for s, p in enumerate(best)],
        })
        with result_cols[i]:
            st.markdown(f"### Trial {i + 1}")
            st.write(f"CO_R = **{co_r:.2f}**, MUT_R = **{mut_r:.2f}**")
            st.dataframe(table, hide_index=True, use_container_width=True,
                         height=670)
            st.metric("Total Ratings", f"{total:.1f}")
        summary.append({"Trial": i + 1, "CO_R": co_r, "MUT_R": mut_r,
                        "Total Ratings": round(total, 2),
                        "Generation best first reached": history.index(max(history))})
        curves[f"Trial {i + 1}"] = history

    st.subheader("Comparison of the 3 trials")
    st.dataframe(pd.DataFrame(summary), hide_index=True)
    st.markdown("Best total rating in each generation:")
    st.line_chart(pd.DataFrame(curves))
