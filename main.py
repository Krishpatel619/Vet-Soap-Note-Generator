import streamlit as st
from gpt4all import GPT4All

# Configure the Streamlit page
st.set_page_config(page_title="Vet SOAP Note Generator (Mistral Instruct)", layout="centered")
st.title("🩺 Veterinary SOAP Note Generator with Mistral Instruct")
st.markdown(
    "Please fill out the form below. The generated SOAP note will use ONLY the provided input and will not invent any details.")

# Input form: all details come directly from the user.
with st.form("clinical_form"):
    species = st.selectbox("Species", ["Dog", "Cat", "Other"])
    breed = st.text_input("Breed", value="Chihuahua")
    age = st.text_input("Age", value="7 years")
    sex = st.selectbox("Sex", ["Male", "Female", "Neutered Male", "Spayed Female"])
    weight = st.text_input("Weight", value="3.5 kg")
    chief_complaint = st.text_area("Chief Complaint", value="Vomiting for 24 hours")
    history = st.text_area("History", value="No known toxin exposure. Normal urination.")
    physical_exam = st.text_area("Physical Exam Findings",
                                 value="T = 38.6°C, HR = 110 bpm, mild cranial abdominal pain")
    submitted = st.form_submit_button("Generate SOAP Note")


def build_prompt(species, breed, age, sex, weight, chief_complaint, history, physical_exam):
    """
    Build a highly specific prompt that instructs the model to output a SOAP note strictly using only the provided input.

    Instructions:
    - Output exactly four sections with the following headers (and nothing else):

      S (Subjective):
      O (Objective):
      A (Assessment):
      P (Plan):

    - Use ONLY the information below.
    - DO NOT invent additional details (e.g., lab results, imaging findings, or numbers) unless explicitly provided.
    - If no data is provided for a section, leave that section blank or write "N/A".
    - Do not output any extra commentary or headings beyond what is specified.

    Patient Input:
      - Species: {species}
      - Breed: {breed}
      - Age: {age}
      - Sex: {sex}
      - Weight: {weight}
      - Chief Complaint: {chief_complaint}
      - History: {history}
      - Physical Exam Findings: {physical_exam}
    """
    prompt = f"""
You are a veterinary assistant tasked with generating a SOAP note using ONLY the following patient input. Do not invent or add any data that is not provided. If a section has no details, output "N/A" for that section.

Output MUST be in the exact format below with four sections (only):

S (Subjective):
[Your Subjective Findings]

O (Objective):
[Your Objective Findings]

A (Assessment):
[Your Assessment]

P (Plan):
[Your Plan]

Patient Input:
- Species: {species}
- Breed: {breed}
- Age: {age}
- Sex: {sex}
- Weight: {weight}
- Chief Complaint: {chief_complaint}
- History: {history}
- Physical Exam Findings: {physical_exam}

Do not include anything else.
"""
    return prompt


# Instantiate the GPT4All model.
# Here we use the Mistral Instruct model. If automatic download fails, manually download the model file
# and specify model_path.
model = GPT4All(model_name="mistral-7b-instruct-v0.1.Q4_0.gguf")


# Uncomment and modify the following line if using a local file:
# model = GPT4All(model_name="mistral-7b-instruct-v0.1.Q4_0.gguf", model_path="/absolute/path/to/mistral-7b-instruct-v0.1.Q4_0.gguf")

def generate_soap_note(prompt):
    """
    Generate a SOAP note using GPT4All with a prediction length of 300 tokens.
    Adjust n_predict if you require longer output.
    """
    response = model.generate(prompt, n_predict=300)
    return response


if submitted:
    with st.spinner("Generating SOAP note..."):
        try:
            prompt = build_prompt(species, breed, age, sex, weight, chief_complaint, history, physical_exam)
            soap_note = generate_soap_note(prompt)
            st.markdown("### 🧾 Generated SOAP Note:")
            st.code(soap_note, language="markdown")
        except Exception as e:
            st.error(f"Error generating SOAP note: {e}")
