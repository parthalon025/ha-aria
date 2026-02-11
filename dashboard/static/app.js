/**
 * HA Intelligence Hub - Dashboard JavaScript
 * Alpine.js helpers and utilities
 */

// Initialize Alpine.js stores
document.addEventListener('alpine:init', () => {
    // WebSocket connection store
    Alpine.store('ws', {
        connected: false,
        message: 'Connecting...'
    });

    // Notification queue
    Alpine.store('notifications', {
        items: [],
        add(message, type = 'info') {
            const id = Date.now();
            this.items.push({ id, message, type });

            // Auto-remove after 3 seconds
            setTimeout(() => {
                this.remove(id);
            }, 3000);
        },
        remove(id) {
            this.items = this.items.filter(item => item.id !== id);
        }
    });
});

// Utility functions
window.dashboardUtils = {
    /**
     * Format timestamp to local string
     */
    formatTimestamp(timestamp) {
        if (!timestamp) return 'N/A';
        return new Date(timestamp).toLocaleString();
    },

    /**
     * Format confidence as percentage
     */
    formatConfidence(confidence) {
        if (confidence === undefined || confidence === null) return 'N/A';
        return `${Math.round(confidence * 100)}%`;
    },

    /**
     * Get confidence class based on value
     */
    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'confidence-high';
        if (confidence >= 0.5) return 'confidence-medium';
        return 'confidence-low';
    },

    /**
     * Get state class based on value
     */
    getStateClass(state) {
        if (!state) return '';
        const stateLower = state.toLowerCase();
        if (stateLower === 'on' || stateLower === 'home') return 'state-on';
        if (stateLower === 'off' || stateLower === 'away') return 'state-off';
        if (stateLower === 'unavailable' || stateLower === 'unknown') return 'state-unavailable';
        return '';
    }
};

// HTMX event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Handle HTMX errors
    document.body.addEventListener('htmx:responseError', (event) => {
        console.error('HTMX error:', event.detail);
        showNotification('Failed to load data', 'error');
    });

    // Handle successful HTMX requests
    document.body.addEventListener('htmx:afterSwap', (event) => {
        console.log('HTMX swap complete:', event.detail);
    });
});
