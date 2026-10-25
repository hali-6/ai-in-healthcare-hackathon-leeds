"""NTC login screen — presentation refresh; authentication unchanged."""
import streamlit as st
import database_aihack as db


def render_login() -> bool:
    """Render NTC login; return True only for authenticated users."""
    if st.session_state.get("user_id"):
        return True

    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      .stApp {background: radial-gradient(ellipse 65% 60% at 95% 5%,rgba(124,49,144,.20),transparent 70%), radial-gradient(ellipse 70% 80% at -10% 100%,rgba(75,34,110,.22),transparent 75%),#110b19;color:#f9f4ff;}
      .stApp, .stApp p, .stApp label, .stApp button, .stApp input {font-family:'DM Sans','Segoe UI',sans-serif;}
      [data-testid="stHeader"] {background:transparent;}
      [data-testid="stSidebar"] {display:none;}
      [data-testid="stMainBlockContainer"], .block-container {max-width:1190px;padding:3.5rem 3.2rem 3rem;}
      .ntc-login-brand {display:flex;align-items:center;justify-content:space-between;gap:1rem;border-bottom:1px solid rgba(232,199,245,.14);padding:0 0 1.65rem;margin-bottom:3.6rem;}
      .ntc-login-wordmark {font:800 1.75rem 'Manrope',sans-serif;letter-spacing:-.085em;color:#fff;}
      .ntc-login-wordmark span {color:#e4a1e9;}
      .ntc-login-brandname {font-size:.72rem;letter-spacing:.24em;text-transform:uppercase;color:#c9b6d3;margin-left:1.1rem;}
      .ntc-login-status {display:inline-flex;align-items:center;gap:.6rem;color:#e5d4ed;font-size:.69rem;letter-spacing:.16em;text-transform:uppercase;white-space:nowrap;}
      .ntc-login-status:before {content:'';width:6px;height:6px;background:#d9a2df;border-radius:50%;box-shadow:0 0 12px rgba(217,162,223,.5);}
      .ntc-login-kicker {font-size:.74rem;letter-spacing:.28em;text-transform:uppercase;color:#e4a1e9;font-weight:700;margin-bottom:1.2rem;}
      .ntc-login-title {font:700 clamp(2.7rem,5vw,4.7rem)/1.08 'Manrope',sans-serif;letter-spacing:-.065em;color:#fff;margin:0 0 1.3rem;}
      .ntc-login-title em {font-style:normal;color:#e4a1e9;}
      .ntc-login-sub {color:#cbbbd6;font-size:1.05rem;line-height:1.8;max-width:30rem;margin:0 0 2rem;}
      .ntc-login-line {height:1px;width:75px;background:linear-gradient(90deg,#e4a1e9,transparent);margin:1.7rem 0 2rem;}
      .ntc-login-value {display:flex;align-items:center;gap:1rem;color:#dfd1e7;font-size:.91rem;margin:1.15rem 0;}
      .ntc-login-value-icon {width:36px;height:36px;display:grid;place-items:center;flex-shrink:0;border-radius:11px;background:rgba(220,161,234,.09);border:1px solid rgba(220,161,234,.2);color:#edb6f0;font-size:1rem;}
      .ntc-login-form-title {font:700 1.7rem 'Manrope',sans-serif;letter-spacing:-.045em;color:#fff;margin:0 0 .55rem;}
      .ntc-login-form-sub {font-size:.92rem;color:#c8b6d2;margin:0 0 1.45rem;line-height:1.6;}
      .stApp div[data-testid="stForm"] {background:linear-gradient(150deg,rgba(48,29,61,.95),rgba(30,19,43,.98));border:1px solid rgba(228,161,233,.22);border-radius:22px;padding:2.25rem 2.3rem 2.15rem;box-shadow:0 30px 75px rgba(0,0,0,.3),inset 0 1px rgba(255,255,255,.045);}
      .stApp div[data-testid="stTextInput"] label p {font-size:.83rem;font-weight:600;color:#f2e8f7;letter-spacing:.02em;}
      .stApp div[data-testid="stTextInput"] input {min-height:3.1rem;background:rgba(15,10,24,.68);color:#fff;border:1px solid rgba(216,175,230,.23);border-radius:10px;padding:0 .95rem;transition:border-color .2s,box-shadow .2s;}
      .stApp div[data-testid="stTextInput"] input:focus {border-color:#e4a1e9;box-shadow:0 0 0 3px rgba(228,161,233,.12);}
      .stApp div[data-testid="stFormSubmitButton"] button {width:100%;min-height:3.35rem;border-radius:11px;background:linear-gradient(110deg,#e6b0ea,#c885da);color:#25112e;border:0;font-weight:800;letter-spacing:.025em;box-shadow:0 12px 26px rgba(154,76,173,.2);transition:transform .2s,filter .2s,box-shadow .2s;}
      .stApp div[data-testid="stFormSubmitButton"] button:hover {color:#25112e;border:0;filter:brightness(1.07);transform:translateY(-1px);box-shadow:0 15px 30px rgba(154,76,173,.29);}
      .ntc-login-caption {font-size:.79rem;color:#ad9cbb;line-height:1.65;margin-top:1.2rem;}
      .ntc-login-footer {border-top:1px solid rgba(232,199,245,.13);padding-top:1.4rem;margin-top:3.5rem;display:flex;justify-content:space-between;gap:1.2rem;color:#a996b8;font-size:.78rem;line-height:1.6;}
      .ntc-login-footer strong {color:#e4c8ec;font-weight:600;}
      @media(max-width:850px) {[data-testid="stMainBlockContainer"],.block-container{padding:2.5rem 1.2rem 2rem;}.ntc-login-brand{margin-bottom:2.3rem}.ntc-login-brandname{display:none}.ntc-login-footer{display:block}.stApp div[data-testid="stForm"]{padding:1.6rem 1.25rem}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="ntc-login-brand"><div><span class="ntc-login-wordmark">NT<span>C</span></span><span class="ntc-login-brandname">NAH TO CANCER</span></div><span class="ntc-login-status">Research prototype</span></div>""", unsafe_allow_html=True)

    intro_col, form_col = st.columns([1.05, 1], gap="large", vertical_alignment="center")
    with intro_col:
        st.markdown("""
        <div class="ntc-login-kicker">EARLY DETECTION · BRIGHTER TOMORROWS</div>
        <h1 class="ntc-login-title">Welcome<br><em>back.</em></h1>
        <p class="ntc-login-sub">A thoughtful approach to skin lesion analysis. Sign in to explore your scans and revisit saved results.</p>
        <div class="ntc-login-line"></div>
        <div class="ntc-login-value"><span class="ntc-login-value-icon">✧</span><span>Purpose-built for research and awareness</span></div>
        <div class="ntc-login-value"><span class="ntc-login-value-icon">▣</span><span>Your previous analyses in one place</span></div>
        <div class="ntc-login-value"><span class="ntc-login-value-icon">◇</span><span>Carefully handled account and image data</span></div>
        """, unsafe_allow_html=True)

    with form_col:
        with st.form("ntc_login_form", clear_on_submit=False):
            st.markdown('<div class="ntc-login-form-title">Account sign in</div><p class="ntc-login-form-sub">Enter your details to continue to NTC.</p>', unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Your username", key="ntc_login_username")
            password = st.text_input("Password", type="password", placeholder="Your password", key="ntc_login_password")
            st.write("")
            submitted = st.form_submit_button("Sign in  →", use_container_width=True)

        if submitted:
            if not username.strip() or not password:
                st.error("Enter your username and password.")
            else:
                try:
                    user = db.authenticate_user(username, password)
                except Exception:
                    # Avoid exposing SQL or infrastructure details on screen.
                    st.error("Sign-in is temporarily unavailable. Please try again later.")
                else:
                    if user is None:
                        st.error("Incorrect username or password.")
                    else:
                        st.session_state["user_id"] = user["user_id"]
                        st.session_state["username"] = user["username"]
                        st.session_state["role"] = user["role"]
                        st.rerun()
        st.markdown('<div class="ntc-login-caption">Need an account? Ask the team to enable registration.<br>Password recovery is not available in this prototype.</div>', unsafe_allow_html=True)

    st.markdown("""<div class="ntc-login-footer"><span><strong>NTC · NAH to Cancer</strong> &nbsp; / &nbsp; Early Detection. Brighter Tomorrows.</span><span>Research and educational prototype only. Not a medical diagnosis or a clinically validated tool. Do not use for medical decisions.</span></div>""", unsafe_allow_html=True)
    return False


def logout() -> None:
    """Clear session authentication state and rerun the app."""
    for key in ("user_id", "username", "role", "ntc_login_password"):
        st.session_state.pop(key, None)
    st.rerun()
