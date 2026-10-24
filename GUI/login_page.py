"""NTC research prototype Streamlit login component.

Import render_login from GUI/gui_main.py. Does not call set_page_config,
so it works inside an existing Streamlit app with its own navigation/sidebar.
"""
import streamlit as st
import database_aihack as db


def render_login() -> bool:
    """Render NTC login; return True only for authenticated users."""
    if st.session_state.get("user_id"):
        return True

    st.markdown("""
    <style>
      :root { --ntc-purple:#da69eb; --ntc-glow:#bd35df; --ntc-pale:#f5d8ff; }
      [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 88% 8%,#261036 0%,transparent 42%),
                    linear-gradient(125deg,#140c1d 0%,#100b18 100%);
        color: #f9f4ff;
      }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] {
        background: linear-gradient(170deg,#30163e,#1b1029 55%,#22122e);
        border-right:1px solid #4b2b5e;
      }
      .block-container { max-width: 1250px; padding-top: 2.2rem; }
      .ntc-eyebrow {font-size: .83rem; font-weight:700; letter-spacing:.28em; color:#f3a5fa;
        border-bottom:1px solid #43234c; padding-bottom:14px; margin-bottom:28px; }
      .ntc-title {font-size: clamp(2.5rem,5vw,4.3rem); letter-spacing:-.045em;
        font-weight:800; line-height:1.07; margin-bottom:10px; }
      .ntc-title span {color:#df70f1;}
      .ntc-subtitle {font-size:1.15rem; color:#dfcbe9; margin-bottom:1.6rem; line-height:1.55;}
      .ntc-panel-head {font-size:2rem; font-weight:750; margin-bottom:2px;}
      .ntc-underline {width:48px;height:2px;background:#df70f1;margin:9px 0 20px;}
      .ntc-muted {color:#d3bfdb; font-size:.96rem;}
      .ntc-security {text-align:center; font-size:4rem; filter:drop-shadow(0 0 18px #ad33d9); padding:1rem 0;}
      .ntc-security-title {font-size:1.55rem;font-weight:750;line-height:1.24;}
      .ntc-security-text {color:#decde7;line-height:1.58;margin-top:13px;}
      .ntc-feature {padding-top:18px; color:#e9d5ef; line-height:1.45;}
      .ntc-feature strong {color:#fff;}
      .ntc-disclaimer {border:1px solid #4e2a63;border-radius:12px;
        background:#1e132a;padding:15px 18px;color:#d7c0e3;font-size:.8rem;margin-top:1rem;}
      div[data-testid="stForm"] {border:1px solid #593068;background:#211329;
        border-radius:16px;padding:24px 28px;}
      div[data-testid="stTextInput"] input {background:#1b1224;color:#fff;
        border:1px solid #634277;border-radius:9px;}
      div[data-testid="stTextInput"] input:focus {border-color:#d96af0;
        box-shadow:0 0 0 2px #9a38ae50;}
      div[data-testid="stFormSubmitButton"] button {width:100%;border:none;
        background:linear-gradient(100deg,#d71ce1,#9239e7);font-weight:750;
        color:white;border-radius:10px;min-height:3.1rem;}
      div[data-testid="stFormSubmitButton"] button:hover {color:white;
        background:linear-gradient(100deg,#e14ef0,#ad55f3);border:none;}
      .ntc-pill {display:inline-block;border:1px solid #71318b;color:#e3a3f4;
        border-radius:22px;padding:7px 16px;font-size:.69rem;letter-spacing:.09em;}
    </style>
    """, unsafe_allow_html=True)

    header_left, header_right = st.columns([4, 1.4])
    with header_left:
        st.markdown('<div class="ntc-eyebrow">ACCOUNT ACCESS</div>', unsafe_allow_html=True)
    with header_right:
        st.markdown('<div class="ntc-pill">⚗ RESEARCH PROTOTYPE</div>', unsafe_allow_html=True)

    st.markdown('<div class="ntc-title">Welcome <span>Back</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="ntc-subtitle">Sign in to access your scan history and saved results.</div>', unsafe_allow_html=True)

    login_col, privacy_col = st.columns([1.2, 1], gap="large")
    with login_col:
        with st.form("ntc_login_form", clear_on_submit=False):
            st.markdown('<div class="ntc-panel-head">Sign In</div><div class="ntc-underline"></div>'
                        '<div class="ntc-muted">Enter your credentials to access your NTC account.</div>',
                        unsafe_allow_html=True)
            st.write("")
            username = st.text_input("Username", placeholder="Enter your username", key="ntc_login_username")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="ntc_login_password")
            submitted = st.form_submit_button("↪  Sign In", use_container_width=True)

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
        st.caption("Need an account? Ask the team to enable registration. Password recovery is not implemented yet.")

    with privacy_col:
        with st.container(border=True):
            st.markdown('<div class="ntc-security">♜</div>', unsafe_allow_html=True)
            st.markdown('<div class="ntc-security-title">Your account and scan data deserve protection</div>', unsafe_allow_html=True)
            st.markdown('<div class="ntc-security-text">NTC uses hashed passwords and encrypts uploaded image data in its local database. This prototype is not approved for real patient data.</div>', unsafe_allow_html=True)
            st.divider()
            st.markdown('<div class="ntc-feature">🛡️ &nbsp; <strong>Protected accounts</strong><br>Passwords are checked against salted hashes.</div>', unsafe_allow_html=True)
            st.markdown('<div class="ntc-feature">🗂️ &nbsp; <strong>Scan history</strong><br>Records are associated with a unique user ID.</div>', unsafe_allow_html=True)
            st.markdown('<div class="ntc-feature">🔒 &nbsp; <strong>Encrypted image storage</strong><br>Images are encrypted before being stored.</div>', unsafe_allow_html=True)

    st.markdown('<div class="ntc-disclaimer">ⓘ NTC is a research and educational prototype, not a clinically validated diagnostic tool. Do not use it to make medical decisions.</div>', unsafe_allow_html=True)
    return False


def logout() -> None:
    """Clear session authentication state and rerun the app."""
    for key in ("user_id", "username", "role", "ntc_login_password"):
        st.session_state.pop(key, None)
    st.rerun()
