(() => {
  "use strict";

  const menuToggle = document.getElementById("menuToggle");
  const mainNav = document.getElementById("mainNav");
  const year = document.getElementById("currentYear");

  if (year) year.textContent = new Date().getFullYear();

  if (menuToggle && mainNav) {
    menuToggle.addEventListener("click", () => {
      const isOpen = mainNav.classList.toggle("open");
      menuToggle.setAttribute("aria-expanded", String(isOpen));
    });

    mainNav.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        mainNav.classList.remove("open");
        menuToggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  // Highlight the navigation item for the section currently in view.
  const navLinks = [...document.querySelectorAll('.main-nav a[href^="#"]')];
  const sections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        navLinks.forEach((link) => {
          link.classList.toggle(
            "active",
            link.getAttribute("href") === `#${entry.target.id}`
          );
        });
      });
    }, { rootMargin: "-30% 0px -60% 0px" });

    sections.forEach((section) => observer.observe(section));
  }
})();
