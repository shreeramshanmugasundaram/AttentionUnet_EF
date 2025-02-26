import streamlit as st 
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt

from actions.videoToFrame import runVideoToFrame
from actions.segmentation import runSegmenation
from actions.selectFrames import runSelectFrames
from actions.calculate import compute_ef

def plot_frame(tensor, frame_index, title):
    """
    Plots a single frame from the tensor.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(tensor[frame_index, :, :], cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    st.pyplot(fig)

# 🎨 Custom CSS for smooth layout transitions and EF styling
st.markdown("""
    <style>
    .uploaded-section {
        transition: width 0.5s ease-in-out;
    }
    .results-section {
        transition: opacity 0.5s ease-in-out, transform 0.5s ease-in-out;
    }
    .ef-box {
        background-color: #f4f4f4;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 📌 Sidebar Upload Section
with st.sidebar:
    st.header("Upload Videos 📤")
    video_file1 = st.file_uploader("Upload 2-Chamber Video", type=["mp4", "avi", "mov", "mkv"])
    video_file2 = st.file_uploader("Upload 4-Chamber Video", type=["mp4", "avi", "mov", "mkv"])

# 🏆 Main Content Area (Selected Frames Appear Here)
if video_file1 and video_file2:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as Video2ch, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as Video4ch:
        
        Video2ch.write(video_file1.read())
        Video4ch.write(video_file2.read())

        # Process videos
        Tensor2ch, Tensor4ch = runVideoToFrame(Video2ch.name, Video4ch.name)
        Channel1_Tensor2ch, Channel1_Tensor4ch = runSegmenation(Tensor2ch[:], Tensor4ch[:])
        dict2ch, dict4ch = runSelectFrames(Channel1_Tensor2ch, Channel1_Tensor4ch)

        # Extract ED and ES frames for EF calculation
        es_mask_2c = Channel1_Tensor2ch[dict2ch["ES_frame"]]
        ed_mask_2c = Channel1_Tensor2ch[dict2ch["ED_frame"]]
        es_mask_4c = Channel1_Tensor4ch[dict4ch["ES_frame"]]
        ed_mask_4c = Channel1_Tensor4ch[dict4ch["ED_frame"]]

        # Compute EF
        ejection_fraction = compute_ef(es_mask_2c, ed_mask_2c, es_mask_4c, ed_mask_4c)  

        # Animate Results Section
        st.markdown('<div class="results-section">', unsafe_allow_html=True)

        st.write("### **End-Diastole (ED) Frames**")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            st.caption("2-Ch ED (Original)")
            plot_frame(Tensor2ch, dict2ch["ED_frame"], "ED - 2-Ch Original")

        with col2:
            st.caption("2-Ch ED (Segmented)")
            plot_frame(Channel1_Tensor2ch, dict2ch["ED_frame"], "ED - 2-Ch Segmented")

        with col3:
            st.caption("4-Ch ED (Original)")
            plot_frame(Tensor4ch, dict4ch["ED_frame"], "ED - 4-Ch Original")

        with col4:
            st.caption("4-Ch ED (Segmented)")
            plot_frame(Channel1_Tensor4ch, dict4ch["ED_frame"], "ED - 4-Ch Segmented")

        st.markdown("---")

        st.write("### **End-Systole (ES) Frames**")
        col5, col6, col7, col8 = st.columns([1, 1, 1, 1])

        with col5:
            st.caption("2-Ch ES (Original)")
            plot_frame(Tensor2ch, dict2ch["ES_frame"], "ES - 2-Ch Original")

        with col6:
            st.caption("2-Ch ES (Segmented)")
            plot_frame(Channel1_Tensor2ch, dict2ch["ES_frame"], "ES - 2-Ch Segmented")

        with col7:
            st.caption("4-Ch ES (Original)")
            plot_frame(Tensor4ch, dict4ch["ES_frame"], "ES - 4-Ch Original")

        with col8:
            st.caption("4-Ch ES (Segmented)")
            plot_frame(Channel1_Tensor4ch, dict4ch["ES_frame"], "ES - 4-Ch Segmented")

        # 💡 Display EF in a highlighted box
        st.markdown(f"""
        <div class="ef-box">
            💓 Ejection Fraction (EF): <span style="color: #d9534f;">{ejection_fraction:.2f}%</span>
        </div>
        </br>
        """, unsafe_allow_html=True)

        # st.success("Process Completed ✅")

        # End animated div
        st.markdown('</div>', unsafe_allow_html=True)

    # Cleanup temp files
    os.remove(Video2ch.name)
    os.remove(Video4ch.name)
