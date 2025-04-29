import streamlit as st
import streamlit.components.v1 as components

# --- Initialize session state for active tab ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

# --- Function to set active tab ---
def set_tab(tab_name):
    st.session_state.active_tab = tab_name

# --- Navbar HTML ---
navbar_html = """
<style>
.navbar {
  background-color: #4CAF50;
  overflow: hidden;
  position: fixed;
  top: 0;
  width: 100%;
  z-index: 999;
}

.navbar a {
  float: left;
  display: block;
  color: #f2f2f2;
  text-align: center;
  padding: 14px 20px;
  text-decoration: none;
  font-size: 17px;
}

.navbar a:hover {
  background-color: #ddd;
  color: black;
}
</style>

<div class="navbar">
  <a href="?tab=Home" target="_self">Home</a>
  <a href="?tab=About" target="_self">About</a>
  <a href="?tab=Services" target="_self">Services</a>
  <a href="?tab=Contact" target="_self">Contact</a>
</div>
"""

# --- Render the navbar ---
components.html(navbar_html, height=60)

# --- Handle tab switching based on URL parameters ---
query_params = st.query_params  # Remove parentheses - this is an object, not a function
if "tab" in query_params:
    set_tab(query_params["tab"][0])

# --- Content Based on Active Tab ---
st.write("")  # Small space because navbar is fixed
st.write("")  

if st.session_state.active_tab == "Home":
    st.title("🏡 Home Page")
    st.write("Welcome to the Home page!")

elif st.session_state.active_tab == "About":
    st.title("ℹ️ About Us")
    st.write("This is the About section.")

elif st.session_state.active_tab == "Services":
    st.title("🛠 Our Services")
    st.write("These are the services we offer.")

elif st.session_state.active_tab == "Contact":
    st.title("📞 Contact Us")
    st.write("Here's how you can reach us.")

else:
    st.title("Page not found 😢")
