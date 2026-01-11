# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in AI Commanders, please report it responsibly.

### How to Report

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting feature
3. Include a detailed description of the vulnerability
4. Provide steps to reproduce if possible

### What to Include

- Description of the vulnerability
- Affected components (e.g., LLM integration, physics engine)
- Potential impact
- Steps to reproduce
- Suggested fix (if any)

### Response Timeline

- We will acknowledge receipt within 48 hours
- We will provide an initial assessment within 7 days
- We will work with you to understand and resolve the issue

## Security Considerations

### API Keys

- Never commit API keys to the repository
- Store your `OPENROUTER_API_KEY` in a `.env` file (which is gitignored)
- Do not share `.env` files or expose API keys in logs

### LLM Integration

- The LLM captains make decisions within a sandboxed simulation
- Captain outputs are parsed through structured tool calls
- No arbitrary code execution from LLM responses

## Best Practices

- Keep dependencies updated
- Review pull requests for security implications
- Use environment variables for sensitive configuration
