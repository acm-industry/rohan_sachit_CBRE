import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-geist-sans)", "ui-sans-serif", "system-ui"],
        mono: ["var(--font-geist-mono)", "ui-monospace", "SFMono-Regular"],
      },
      colors: {
        accent: {
          DEFAULT: "#7c3aed", // violet-600
          50: "#f5f3ff",
          400: "#a78bfa",
          500: "#8b5cf6",
          600: "#7c3aed",
          700: "#6d28d9",
        },
      },
      keyframes: {
        pulseSoft: {
          "0%, 100%": { opacity: "0.85" },
          "50%": { opacity: "1" },
        },
        glow: {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(139, 92, 246, 0.0)" },
          "50%": { boxShadow: "0 0 0 6px rgba(139, 92, 246, 0.18)" },
        },
      },
      animation: {
        pulseSoft: "pulseSoft 1.2s ease-in-out infinite",
        glow: "glow 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
