/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'iris-bg':        '#08080f',
        'iris-surface':   '#0f0f1a',
        'iris-elevated':  '#161625',
        'iris-border':    '#1e1e35',
        'iris-accent':    '#7c3aed',
        'iris-cyan':      '#06b6d4',
        'iris-success':   '#22c55e',
        'iris-danger':    '#ef4444',
        'iris-text':      '#f1f1f5',
        'iris-secondary': '#8b8ba7',
        'iris-muted':     '#4a4a6a',
      },
      fontFamily: {
        inter: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'fade-in':    'fadeIn 0.3s ease forwards',
        'slide-up':   'slideUp 0.2s ease forwards',
      },
      keyframes: {
        fadeIn:  { from: { opacity: '0' },                   to: { opacity: '1' } },
        slideUp: { from: { opacity: '0', transform: 'translateY(8px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
      },
    },
  },
  plugins: [],
}
