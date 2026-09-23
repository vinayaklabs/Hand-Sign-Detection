(() => {
    const savedTheme = localStorage.getItem("signspeak-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const initialTheme = savedTheme || (prefersDark ? "dark" : "light");

    document.documentElement.dataset.theme = initialTheme;

    const currentPath = window.location.pathname.replace(/\/$/, "") || "/";
    const pageRoutes = {
        HOME: "/",
        ABOUT: "/about",
        "CONTACT US": "/contact-us",
        "SIGN IN": "/sign-in",
        LOGIN: "/login"
    };

    document.querySelectorAll(".nav-buttons button, .auth-buttons button").forEach((button) => {
        const label = button.textContent.trim().replace(/\s+/g, " ");
        button.classList.toggle("active", pageRoutes[label] === currentPath);
    });

    function updateToggle(toggle) {
        const isDark = document.documentElement.dataset.theme === "dark";
        toggle.textContent = isDark ? "☀" : "☾";
        toggle.setAttribute("aria-label", isDark ? "Switch to light theme" : "Switch to dark theme");
        toggle.setAttribute("title", isDark ? "Switch to light theme" : "Switch to dark theme");
    }

    document.querySelectorAll(".theme-toggle").forEach((toggle) => {
        updateToggle(toggle);
        toggle.addEventListener("click", () => {
            const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
            document.documentElement.dataset.theme = nextTheme;
            localStorage.setItem("signspeak-theme", nextTheme);
            updateToggle(toggle);
        });
    });
})();
