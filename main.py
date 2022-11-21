# %%
import Snapshot_code
import pandas as pd
import streamlit as st
import base64
from zipfile import ZipFile
from io import BytesIO
def main():
    st.title("SNAPSHOT TOOL")
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:

        new_df2 = pd.read_csv(uploaded_file)
        st.write(new_df2)


    mapping_dict = {'Baseline':'US_Population','Segment_1':'Application Completed'}

    save_path = 'D:/Job App/'
    # Set up variables for snapshot
    file_name = 'Unlock_11182022_Application Completed_1108'
    seg_var = 'category'
    epsilon_path = 'Data/Epsilon_Final.xlsx'
    bin_vars_path = 'Data/Epsilon_attributes_binning_2.csv'
    # Read in file and set bins
    text_contents = '''This is some text'''
    st.download_button('Download some text', text_contents)
    
    if st.button("Run Code"):
        df,data1 = Snapshot_code.Snapshot_Profile(profile_data = new_df2, segment_var=seg_var, continuous_path = bin_vars_path,segments = None, segment_names = None, include = None,variable_order = None, other_segment = False,
                        file = file_name, exclude = [], PPT = True, continuous = [], excludeother = False,mapping_dict=mapping_dict,save_path=save_path)
        
       


        data1.save('temp1.xlsx')
        zipObj = ZipFile("sample.zip", "w")
        # Add multiple files to the zip
        zipObj.write("temp1.xlsx")
        # close the Zip File
        zipObj.close()

        ZipfileDotZip = "sample.zip"

        with open(ZipfileDotZip, "rb") as f:
            bytes = f.read()
            b64 = base64.b64encode(bytes).decode()
            href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}.zip'>\
                Click last model weights\
            </a>"
        st.sidebar.markdown(href, unsafe_allow_html=True)

if __name__=='__main__':
    main()
