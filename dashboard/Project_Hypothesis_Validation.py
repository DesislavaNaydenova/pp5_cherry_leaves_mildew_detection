import streamlit as st
import matplotlib.pyplot as plt

def Project_Hypothesis_Validation_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* We hypothesize that cherry leaves affected by mildew will show clear signs, "
        f"such as white powdery spots or discoloration, that differentiate them from healthy leaves. \n\n"
        f"* Image analyses, such as average image, variability image, and differences between healthy and "
        f"mildew-affected leaves, are expected to reveal patterns or distinctive features that can help "
        f"distinguish leaves with mildew from healthy ones."
    )

    st.write("### Visual Hypothesis Testing")

    # Here you can insert images to show the differences you hypothesize.
    st.image(
        "assets/images/healthy/2a6787af-106b-4d86-b758-35589b2cfab3___JR_HL 9850.JPG", caption="Healthy Cherry Leaf",
        use_column_width=True
    )
    st.image(
        "assets/images/powdery_mildew/00e0a4ab-ecbd-4560-a71c-b19d86bb087c___FREC_Pwd.M 4917_flipLR.JPG", caption="Cherry Leaf with Mildew",
        use_column_width=True
    )

    st.write("### Conclusion")

    st.success(
        f"From the analysis and visual inspection, we can conclude that there are distinguishable "
        f"features between healthy and mildew-affected cherry leaves, validating the hypothesis. "
        f"The trained model uses these features to predict the presence of mildew accurately."
    )
