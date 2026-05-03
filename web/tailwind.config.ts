import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: "#005BC4",
          light: "#38bdf8",
          dark: "#003d8a",
        },
        surface: {
          DEFAULT: "#f6f9ff",
          card: "#ffffff",
          ink: "#0f172a",
          muted: "#64748b",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
      },
      boxShadow: {
        soft: "0 10px 30px rgba(2, 6, 23, 0.06)",
        glow: "0 8px 22px rgba(0, 91, 196, 0.35)",
      },
    },
  },
  plugins: [],
};

export default config;
