#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import Draw
from PIL import Image
import py3Dmol
st.markdown("<h1 style='text-align: center; color: darkblue;'>RNS Reaction Rate Predictor</h1>", unsafe_allow_html=True)
st.markdown("Enter the compound SMILES, temperature, and select the radical type to predict the log(k) value of its reaction with the radical.")
col1, col2= st.columns(2)
with col1:
    smiles = st.text_input("SMILES:")
with col2:
    temperature = st.number_input("Temperature (K)", value=300.0)
radical_options = {
    "NO₂·": "NO2",
    "NH₂·": "NH2"
}
radical_display = st.radio(
    "Radical Type",
    options=list(radical_options.keys()),
    horizontal=True
)
radical = radical_options[radical_display]


# In[ ]:


ph_scaler=joblib.load('ph_scaler.pkl')
temperature_scaler=joblib.load('temperature_scaler.pkl')
xgb_maccs=joblib.load('xgb_bo.pkl')
def model(smiles,temperature,radical):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    maccs=MACCSkeys.GenMACCSKeys(mol)
    maccs=np.array(maccs)
    temperature=temperature_scaler.transform([[temperature]])[0]
    pH=ph_scaler.transform([[7]])[0]
    if radical == "NO2":
        type=np.array([0,1])
    else:
        type=np.array([1,0])
    maccs_sample=np.concatenate((maccs,type,temperature,pH),axis=0).reshape(1,-1)
    pre_maccs=xgb_maccs.predict(maccs_sample)[0]
    return pre_maccs
def render_3d_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    mol_block = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=400, height=400)
    view.addModel(mol_block, "mol")
    view.setStyle({'stick': {}})
    view.setBackgroundColor('0xeeeeee')
    view.zoomTo()
    return view
if st.button("Predict"):
    if smiles.strip()=='':
        st.error("⚠️ Please enter a valid SMILES string.")
    else:
        a=model(smiles,temperature,radical)
        if a is None:
            st.error('⚠️ Failed to parse SMILES. Please check the structure.')
        else:
            st.markdown("### 📊 Predicted log(k) Values:")
            st.success(f"{a:.4f}")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.markdown("### 🧬 Molecular Structure Visualization")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🖼️ 2D Structure")
                    img = Draw.MolToImage(mol, size=(300, 300))
                    st.image(img)
                with col2:
                    st.markdown("#### 🔬 3D Structure (Interactive)")
                    view = render_3d_molecule(smiles)
                    view_html = view._make_html()
                    st.components.v1.html(view_html, height=450, width=450)


# In[ ]:


st.markdown("""
    <style>
    body {
        background-color: #f0f8ff;
        font-family: 'Arial', sans-serif;
    }

    .stButton button {
    background-color: #4CAF50;
    color: white !important;
    border-radius: 12px;
    padding: 14px 28px;
    font-size: 20px;
    font-weight: bold;
    border: none;
    transition: 0.3s ease;
}

    .stButton button:hover {
        background-color: #45a049;
        cursor: pointer;
}
    .stButton button:active {
        color: white !important;
        background-color: #3e8e41;
}
    .stTextInput input, .stNumberInput input {
        background-color: #ffffff;
    }

    .main {
        max-width: 800px;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)


# In[ ]:




