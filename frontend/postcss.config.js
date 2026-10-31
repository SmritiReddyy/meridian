// Without this, Next.js serves globals.css as-is and the @tailwind directives
// never expand, so the whole UI renders unstyled.
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
