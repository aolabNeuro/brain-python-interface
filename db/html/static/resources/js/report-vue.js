/**
 * Vue.js based Report Component
 * Replaces the jQuery-based Report class with a reactive Vue component
 * 
 * Benefits:
 * - Automatic UI updates when data changes
 * - Cleaner state management
 * - Less boilerplate code
 */

// Create the Vue app for the report section
const reportApp = {
    el: '#report_div',
    data() {
        return {
            reportStats: {},
            messages: [],
            stdoutText: '',
            wsConnected: false,
            ws: null,
            reconnectAttempts: 0,
            maxReconnectAttempts: 10,
        };
    },
    computed: {
        // Convert reportStats object to array for easy display
        statsArray() {
            return Object.entries(this.reportStats)
                .filter(([key, value]) => {
                    // Hide these specific stats
                    return !key.match(/rig_name|status|task|subj|date|idx/);
                })
                .map(([key, value]) => ({
                    name: key,
                    value: value
                }));
        },
        // Track FPS for statistics
        fpsList() {
            return this.statsArray
                .filter(stat => stat.name.toLowerCase().includes('fps'))
                .map(stat => stat.value);
        }
    },
    methods: {
        /**
         * Initialize WebSocket connection
         */
        initWebSocket() {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                return; // Already connected
            }

            var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            var host = window.location.host;
            const wsUrl = protocol + "//" + host + "/ws/tasks/";

            try {
                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log("Report WebSocket connected");
                    this.wsConnected = true;
                    this.reconnectAttempts = 0;
                };

                this.ws.onmessage = (evt) => {
                    try {
                        const data = JSON.parse(evt.data);
                        
                        if (data.type === 'task_status') {
                            // Update report stats - Vue will automatically re-render
                            this.reportStats = data.reportstats || {};
                        } else if (data.type === 'error') {
                            this.addMessage(`Error: ${data.message}`, 'error');
                        } else if (data.type === 'stdout') {
                            this.stdoutText += data.message + '\n';
                        }
                    } catch (e) {
                        console.error("Error processing WebSocket message:", e);
                    }
                };

                this.ws.onerror = (error) => {
                    console.error("Report WebSocket error:", error);
                    this.wsConnected = false;
                };

                this.ws.onclose = () => {
                    console.log("Report WebSocket disconnected");
                    this.wsConnected = false;
                    this.attemptReconnect();
                };
            } catch (e) {
                console.error("Failed to create WebSocket:", e);
                this.attemptReconnect();
            }
        },

        /**
         * Attempt to reconnect with exponential backoff
         */
        attemptReconnect() {
            if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                console.error("Max reconnection attempts reached");
                return;
            }

            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            this.reconnectAttempts++;
            
            console.log(`Attempting to reconnect WebSocket in ${delay}ms...`);
            setTimeout(() => {
                this.initWebSocket();
            }, delay);
        },

        /**
         * Manually update report stats (fallback for AJAX)
         */
        manualUpdate() {
            $.post('report', {}, (info) => {
                this.reportStats = info.data || {};
            });
        },

        /**
         * Add a message to the report
         */
        addMessage(message, type = 'info') {
            const timestamp = new Date().toLocaleTimeString();
            this.messages.push({
                timestamp,
                type,
                message
            });
            
            // Keep only last 100 messages
            if (this.messages.length > 100) {
                this.messages.shift();
            }
        },

        /**
         * Clear stdout text
         */
        clearStdout() {
            this.stdoutText = '';
        },

        /**
         * Save FPS log
         */
        saveFpsLog() {
            if (this.fpsList.length === 0) {
                alert("No FPS data to save");
                return;
            }

            const blob = new Blob([this.fpsList.join("\n")], { type: "text/plain" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "fps_log.txt";
            a.click();
            URL.revokeObjectURL(url);
        },

        /**
         * Activate the report (called when task starts)
         */
        activate() {
            console.log("Activating report...");
            this.initWebSocket();
        },

        /**
         * Deactivate the report (called when task stops)
         */
        deactivate() {
            if (this.ws) {
                this.ws.close();
                this.ws = null;
            }
            this.wsConnected = false;
        }
    },

    mounted() {
        console.log("Report Vue app mounted");
        // Don't auto-connect, wait for activate() to be called
    },

    beforeUnmount() {
        this.deactivate();
    }
};

// Export for use in list.js
window.reportVueApp = reportApp;
