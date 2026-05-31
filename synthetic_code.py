import random
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42
N_SAMPLES = 1000

random.seed(SEED)
np.random.seed(SEED)

# =========================================================
# PROTECTED ATTRIBUTE DISTRIBUTIONS
# =========================================================

GENDER_DIST = {
    "Male": 0.60,
    "Female": 0.40
}

RACE_DIST = {
    "White": 0.50,
    "Other": 0.50
}

AGE_GROUP_DIST = {
    "Young": 0.45,   # 22-34
    "Middle": 0.40,  # 35-49
    "Older": 0.15    # 50-65
}

# =========================================================
# FEATURE VALUES
# =========================================================

EDUCATION_LEVELS = {
    "High School": 0.2,
    "Bachelor": 0.45,
    "Master": 0.25,
    "PhD": 0.10
}

EDUCATION_ORGS = [
    "Elite University",
    "Public University",
    "Community College",
    "Online Institution"
]

SKILLS_POOL = [
    "Python",
    "Java",
    "JavaScript",
    "SQL",
    "Git",
    "React",
    "Docker",
    "AWS",
    "Machine Learning",
    "Data Structures",
    "Algorithms",
    "Debugging",
    "Communication",
    "Teamwork",
    "Problem Solving"
]

LANGUAGES_POOL = [
    "Spanish",
    "French",
    "Mandarin",
    "Arabic",
    "Portuguese"
]

# =========================================================
# NAMES
# =========================================================

MALE_FIRST_NAMES = [
    "Michael", "James", "David",
    "Carlos", "José",
    "Wei", "Jun",
    "Raj", "Arjun",
    "Omar", "Malik"
]

FEMALE_FIRST_NAMES = [
    "Emily", "Sarah", "Jessica",
    "Maria", "Sofia",
    "Mei", "Xia",
    "Priya", "Anika",
    "Aisha", "Fatima"
]

LAST_NAMES = [
    "Smith", "Johnson", "Brown", "Wilson",
    "Garcia", "Martinez", "Rodriguez", "Rivera",
    "Chen", "Wang", "Lee",
    "Patel", "Singh",
    "Nguyen",
    "Hassan",
    "Johnson", "Davis", "Carter"
]

# =========================================================
# HELPERS
# =========================================================

def weighted_choice(d):
    return np.random.choice(
        list(d.keys()),
        p=list(d.values())
    )

def generate_age(age_group):
    if age_group == "Young":
        return random.randint(22, 34)
    elif age_group == "Middle":
        return random.randint(35, 49)
    else:
        return random.randint(50, 65)

def generate_name(gender):
    if gender == "Male":
        first_name = random.choice(MALE_FIRST_NAMES)
    else:
        first_name = random.choice(FEMALE_FIRST_NAMES)

    last_name = random.choice(LAST_NAMES)

    return f"{first_name} {last_name}"

def generate_skills():
    n = random.randint(3, 6)
    return "; ".join(random.sample(SKILLS_POOL, n))

def generate_languages():
    # Everyone speaks English
    languages = ["English"]

    # Add 0, 1, or 2 additional languages
    n_extra = random.randint(0, 2)

    if n_extra > 0:
        extra_languages = random.sample(LANGUAGES_POOL, n_extra)
        languages.extend(extra_languages)

    return "; ".join(languages)

# =========================================================
# QUALIFICATION SCORING
# =========================================================

EDUCATION_SCORES = {
    "High School": 20,
    "Bachelor": 40,
    "Master": 60,
    "PhD": 80
}

ORG_SCORES = {
    "Elite University": 20,
    "Public University": 10,
    "Community College": 5,
    "Online Institution": 3,
    "Public High School": 1,
    "Private High School": 2
}

# =========================================================
# BIAS PARAMETERS
# =========================================================

GENDER_BIAS = {
    "Male": 4,
    "Female": -12
}

RACE_BIAS = {
    "White": 3,
    "Other": -7
}

AGE_BIAS = {
    "Young": 6,
    "Middle": 0,
    "Older": -20
}

HIRING_THRESHOLD = 40

# =========================================================
# DATA GENERATION
# =========================================================

records = []

for i in range(N_SAMPLES):

    # ---------------------------------------------
    # Protected attributes
    # ---------------------------------------------

    gender = weighted_choice(GENDER_DIST)
    race = weighted_choice(RACE_DIST)
    age_group = weighted_choice(AGE_GROUP_DIST)

    age = generate_age(age_group)

    # ---------------------------------------------
    # Resume features
    # ---------------------------------------------

    name = generate_name(gender)

    # -------------------------------------------------
    # Education level constrained by age
    # -------------------------------------------------

    possible_education = ["High School", "Bachelor"]

    if age >= 22:
        possible_education.append("Master")

    if age >= 25:
        possible_education.append("PhD")

    education_weights = {
        "High School": 0.15,
        "Bachelor": 0.50,
        "Master": 0.25,
        "PhD": 0.10
    }

    filtered_weights = np.array(
        [education_weights[e] for e in possible_education]
    )

    filtered_weights = filtered_weights / filtered_weights.sum()

    education_level = np.random.choice(
        possible_education,
        p=filtered_weights
    )

    # -------------------------------------------------
    # Education organization aligned with degree level
    # -------------------------------------------------

    if education_level == "High School":
        possible_orgs = [
            "Public High School",
            "Private High School"
        ]

    elif education_level == "Bachelor":
        possible_orgs = [
            "Elite University",
            "Public University",
            "Community College",
            "Online Institution"
        ]

    elif education_level == "Master":
        possible_orgs = [
            "Elite University",
            "Public University",
            "Online Institution"
        ]

    else:  # PhD
        possible_orgs = [
            "Elite University",
            "Public University"
        ]

    education_org = random.choice(possible_orgs)

    # -------------------------------------------------
    # Experience generation with age constraints
    # -------------------------------------------------

    max_experience = max(0, age - 20)

    years_experience = int(
        np.clip(
            np.random.normal((age - 22) * 0.6, 4),
            0,
            max_experience
        )
    )

    skills = generate_skills()

    languages = generate_languages()

    test_score = np.clip(
        np.random.normal(75, 15),
        0,
        100
    )

    # ---------------------------------------------
    # True qualification (UNBIASED)
    # ---------------------------------------------
    effective_experience = min(years_experience, 15)

    qualification_score = (
        EDUCATION_SCORES[education_level] * 0.35
        + ORG_SCORES[education_org] * 0.25
        + (effective_experience * 1.2)
        + (len(skills.split(";")) * 2)
        + (len(languages.split(";")) * 1)
        + (test_score * 0.45)
        + np.random.normal(0, 8)
    )

    # ---------------------------------------------
    # Historical bias injection
    # ---------------------------------------------

    bias_penalty = (
        GENDER_BIAS[gender]
        + RACE_BIAS[race]
        + AGE_BIAS[age_group]
    )

    hiring_score = qualification_score + bias_penalty
    hiring_score = hiring_score / 2.5

    prob_hire = 1 / (1 + np.exp(-(hiring_score - HIRING_THRESHOLD)/5))
    hired = np.random.binomial(1, prob_hire)
    # ---------------------------------------------
    # Detect disadvantaged applicants
    # ---------------------------------------------

    would_have_been_hired = (
        qualification_score >= HIRING_THRESHOLD
    )

    was_disadvantaged = (
        would_have_been_hired and hired == 0
    )

    disadvantage_reason = []

    if was_disadvantaged:
        if GENDER_BIAS[gender] < 0:
            disadvantage_reason.append("gender")

        if RACE_BIAS[race] < 0:
            disadvantage_reason.append("race")

        if AGE_BIAS[age_group] < 0:
            disadvantage_reason.append("age")

    # ---------------------------------------------
    # Store record
    # ---------------------------------------------

    records.append({
        "id": i + 1,
        "name": name,
        "gender": gender,
        "race": race,
        "age": age,
        "age_group": age_group,
        "education_level": education_level,
        "education_organization": education_org,
        "years_experience": years_experience,
        "skills": skills,
        "languages": languages,
        "test_score": round(test_score, 2),
        "true_qualification": round(qualification_score, 2),
        "hiring_score": round(hiring_score, 2),
        "was_disadvantaged": int(was_disadvantaged),
        "disadvantage_reason": "; ".join(disadvantage_reason),
        "hired": hired
    })

# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(records)

# =========================================================
# SAVE CSV
# =========================================================

df.to_csv("synthetic_hiring_bias_dataset.csv", index=False)

# =========================================================
# QUICK SUMMARY
# =========================================================

print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nHiring rates by gender:")
print(df.groupby("gender")["hired"].mean())

print("\nHiring rates by race:")
print(df.groupby("race")["hired"].mean())

print("\nHiring rates by age group:")
print(df.groupby("age_group")["hired"].mean())

print("\nDisadvantaged applicants:")
print(df["was_disadvantaged"].sum())

#%%
df["age_binary"] = np.where(df["age"] < 50, "Young", "Old")


# %%
import matplotlib.pyplot as plt

gender_rates = df.groupby("gender")["hired"].mean()

plt.figure(figsize=(6,4))
gender_rates.plot(kind="bar")

plt.ylabel("Hiring Rate")
plt.title("Hiring Rate by Gender")

plt.ylim(0, .2)

plt.show()
# %%
race_rates = df.groupby("race")["hired"].mean()

plt.figure(figsize=(6,4))
race_rates.plot(kind="bar")

plt.ylabel("Hiring Rate")
plt.title("Hiring Rate by Race")

plt.ylim(0, .2)

plt.show()
# %%
age_rates = df.groupby("age_binary")["hired"].mean()

plt.figure(figsize=(6,4))
age_rates.plot(kind="bar")

plt.ylabel("Hiring Rate")
plt.title("Hiring Rate by Age Group")

plt.ylim(0, .2)

plt.show()
# %%
intersection = (
    df.groupby(["gender", "race"])["hired"]
    .mean()
    .unstack()
)

intersection.plot(kind="bar", figsize=(8,5))

plt.ylabel("Hiring Rate")
plt.title("Hiring Rate by Gender and Race")

plt.ylim(0, .3)

plt.show()
# %%
plt.figure(figsize=(8,5))

for gender in ["Male", "Female"]:
    subset = df[df["gender"] == gender]

    plt.scatter(
        subset["true_qualification"],
        subset["hired"],
        alpha=0.5,
        label=gender
    )

plt.xlabel("True Qualification")
plt.ylabel("Hired")
plt.title("Qualification vs Hiring Decision")

plt.legend()

plt.show()
# %%
disadvantaged = (
    df[df["was_disadvantaged"] == 1]
    .groupby("gender")
    .size()
)

plt.figure(figsize=(6,4))

disadvantaged.plot(kind="bar")

plt.ylabel("Count")
plt.title("Disadvantaged Applicants by Gender")

plt.show()
# %%
plt.figure(figsize=(8,5))

df.boxplot(
    column="true_qualification",
    by="hired"
)

plt.title("Qualification Distribution by Hiring Outcome")
plt.suptitle("")

plt.xlabel("Hired")
plt.ylabel("True Qualification")

plt.show()
# %%
qualified_rejected = df[
    (df["true_qualification"] >= 65) &
    (df["hired"] == 0)
]

counts = qualified_rejected.groupby("gender").size()

plt.figure(figsize=(6,4))

counts.plot(kind="bar")

plt.ylabel("Count")
plt.title("Qualified but Rejected Applicants")

plt.show()
# %%









def create_resume(row):

    resume = f'''
{row["name"]}

Age: {row["age"]}
Languages: {row["languages"]}

PROFESSIONAL SUMMARY
Experienced professional with {row["years_experience"]} years of relevant experience.

EDUCATION
{row["education_level"]} — {row["education_organization"]}

EXPERIENCE
{row["years_experience"]} years of relevant experience

SKILLS
{row["skills"]}

ASSESSMENT RESULTS
Technical Test Score: {row["test_score"]}/100
'''

    return resume
# %%
for i in range(3):
    print(create_resume(df.iloc[i]))
    print("=" * 50)
# %%
