/**
 * Vue.js based Task Controls Component
 * Manages task start/stop buttons and state
 * 
 * Replaces jQuery event handlers with Vue reactive methods
 */

const taskControlsApp = {
    el: '#task_controls_vue',
    data() {
        return {
            taskRunning: false,
            taskState: 'idle',  // idle, running, testing, error, stopped
            selectedMode: 'start',  // 'test', 'start', 'saverec'
        };
    },
    computed: {
        // Show start buttons when no task is running
        showStartButtons() {
            return !this.taskRunning;
        },
        // Show stop button when task is running
        showStopButtons() {
            return this.taskRunning;
        },
        // Button labels and descriptions
        startButtonLabel() {
            return this.selectedMode === 'test' ? 'Test' :
                   this.selectedMode === 'saverec' ? 'Save Record' :
                   'Start Experiment';
        },
        startButtonTitle() {
            return this.selectedMode === 'test' ? 'Run the experiment without saving any data' :
                   this.selectedMode === 'saverec' ? 'Save a record of an already complete experiment' :
                   'Start the experiment with automatic saving';
        }
    },
    methods: {
        /**
         * Start experiment with selected mode
         */
        startExperiment() {
            if (this.selectedMode === 'test') {
                this.executeStart('test');
            } else if (this.selectedMode === 'saverec') {
                this.executeStart('saverec');
            } else {
                this.executeStart('start');
            }
        },

        /**
         * Execute start via jQuery (maintain compatibility with existing code)
         */
        executeStart(mode) {
            const btn = mode === 'test' ? '#testbtn' :
                       mode === 'saverec' ? '#saverecbtn' :
                       '#startbtn';
            $(btn).click();
        },

        /**
         * Stop experiment
         */
        stopExperiment() {
            $('#stopbtn').click();
        },

        /**
         * Update task state based on interface state changes
         */
        updateTaskState(newState) {
            this.taskState = newState;
            this.taskRunning = ['running', 'testing'].includes(newState);
        },

        /**
         * Set the experiment mode
         */
        setMode(mode) {
            this.selectedMode = mode;
        }
    },

    mounted() {
        console.log("Task Controls Vue app mounted");
        // Don't start anything yet
    }
};

// Export for use in list-vue.js
window.taskControlsApp = taskControlsApp;

/**
 * Bridge function to update Vue state from jQuery event handlers
 * Called from list-vue.js when task state changes
 */
window.updateVueTaskState = function(newState) {
    if (taskControlsApp.instance && taskControlsApp.instance.updateTaskState) {
        taskControlsApp.instance.updateTaskState(newState);
    }
};
