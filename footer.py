import streamlit as st

def show_footer():
    footer_html = """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            height: 60px;
            background-color: black;
            color: white;
            text-align: center;
            padding: 10px 0;
        }
    </style>
    <div class="footer">
        <div>&copy; All rights reserved.</div>
        <div>CopyRight &copy; 2024 Team CodeCrafter (Govind, Rahul, Prashant)</div>
    </div>
    """

    # Render the footer HTML
    st.markdown(footer_html, unsafe_allow_html=True)

# Call the function to display the footer
# show_footer()
