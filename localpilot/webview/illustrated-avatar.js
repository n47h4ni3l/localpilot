// illustrated-avatar.js
// This module provides a simple function to create an illustrated avatar
// without performing any page navigation. It is intentionally minimal to
// satisfy strict-CSP and static analysis requirements.

/**
 * Creates an illustrated avatar element.
 * @param {string} name - The name to display on the avatar.
 * @param {string} [color='#3498db'] - Optional background color.
 * @returns {HTMLElement} The avatar element.
 */
export function createAvatar(name, color = '#3498db') {
  const avatar = document.createElement('div');
  avatar.style.width = '100px';
  avatar.style.height = '100px';
  avatar.style.borderRadius = '50%';
  avatar.style.backgroundColor = color;
  avatar.style.display = 'flex';
  avatar.style.alignItems = 'center';
  avatar.style.justifyContent = 'center';
  avatar.style.fontSize = '36px';
  avatar.style.color = '#fff';
  avatar.style.userSelect = 'none';
  avatar.textContent = name.charAt(0).toUpperCase();
  return avatar;
}

// Example usage (uncomment to test in a browser environment):
// const container = document.getElementById('avatar-container');
// if (container) {
//   container.appendChild(createAvatar('Alice', '#e74c3c'));
// }
