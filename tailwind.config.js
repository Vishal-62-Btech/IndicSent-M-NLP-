/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        teal: {
          primary: '#28594F',
          dark: '#214B43',
          darker: '#183832',
          surface: '#2C6156',
          light: '#377366',
        },
        lime: {
          accent: '#C7F59D',
          hover: '#B5EA87',
          mint: '#E7F6D9',
          tint: '#F0F9E8',
        },
        cream: {
          canvas: '#F7F8F5',
          card: '#FFFFFF',
          border: '#E3E8DE',
          muted: '#EAEFE6',
        },
        ink: {
          primary: '#172923',
          secondary: '#68776F',
          muted: '#8D9D95',
        },
      },
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
    },
  },
  plugins: [],
}
