import streamlit as st
import matplotlib.pyplot as plt


def Quick_Project_Summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n"
        f"* Malaria is a parasitic infection transmitted by the bite of infected female "
        f"Anopheles mosquitoes.\n")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/GyanShashwat1611/WalkthroughProject01/blob/main/README.md).")
    

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in having a study to differentiate "
        f"a parasitized and uninfected cell visually.\n"
        f"* 2 - The client is interested to tell whether a given cell contains malaria parasite or not. "
        )