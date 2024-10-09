import streamlit as st

import streamlit as st

def set_page_logo(logo_path):
    st.set_page_config(page_title="LucidLens Platform", page_icon=logo_path)


def NavForAll():
    # st.image("/home/govind/Documents/MLPR/final/logo.png", width=50)


    st.header('', divider='rainbow')

    st.write("")
    st.write("")
    st.write("")
    st.write("")




    st.write("""
    <style>
        .box:hover {
            background-color: lightblue;
        }
        .dropdown {
            position: relative;
            display: inline-block;
        }
        .dropdown-content {
            display: none;
            position: absolute;
            background-color: white;
            min-width: 220px;
            color: black;
            text-decoration: none;
            box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
            z-index: 1;
        }
        .dropdown:hover .dropdown-content {
            display: block;
        }
        .dropdown-content a {
            padding: 12px 16px;
            display: block;
            text-decoration:none;
        }
        .dropdown-content a:hover {
            background-color: lightblue;
        }
        .next {
            display: grid;
            grid-template-columns: repeat(2, 100px);
            gap: 10px;
        }
    </style>
    
    <div style="height: 5vh;">
        <div id="my-element" style="color: white; width: 100%; background-color: grey; height: 50px; display: flex; justify-content: space-between; align-items: center; border: 2px solid black; border-radius: 5px;">
            <div class="dropdown" style="padding-left: 40px;">Other Products
                <div class="dropdown-content">
                    <a href="">Smart Googles</a>
                    <a href="">Vision Stick</a>
                    <a href="">Download APP</a>
                    <a href="">Smart Watches</a>                                   
             </div>
            </div>
            <div style="display: flex; flex-direction: row; padding-right: 20px; justify-content: center;">
                <div class="box" style="margin-left:20px;padding: 10px;border-radius:9px">Home</div>
                <div class="box" style="margin-left:20px;padding: 10px;border-radius:9px">About us</div>
                <div class="box" style="margin-left:20px;padding: 10px;border-radius:9px">Product info</div>
                <div class="box" style="margin-left:20px;padding: 10px;border-radius:9px">Login</div>
                <div class="box" style="margin-left:20px;padding: 10px;border-radius:9px">Contact Us</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
