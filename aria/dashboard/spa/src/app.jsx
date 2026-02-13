import { Component } from 'preact';
import { useEffect } from 'preact/hooks';
import Router from 'preact-router';
import { connectWebSocket, disconnectWebSocket } from './store.js';
import Sidebar from './components/Sidebar.jsx';
import ErrorState from './components/ErrorState.jsx';
import Home from './pages/Home.jsx';
import Discovery from './pages/Discovery.jsx';
import Capabilities from './pages/Capabilities.jsx';
import Predictions from './pages/Predictions.jsx';
import Patterns from './pages/Patterns.jsx';
import Automations from './pages/Automations.jsx';
import Intelligence from './pages/Intelligence.jsx';
import Shadow from './pages/Shadow.jsx';
import Settings from './pages/Settings.jsx';
import DataCuration from './pages/DataCuration.jsx';

/**
 * Custom hash-based history for preact-router.
 * Converts hash URLs (#/path) into the pathname-based API that preact-router expects.
 */
function createHashHistory() {
  const listeners = [];

  function getLocation() {
    const hash = window.location.hash || '#/';
    const path = hash.replace(/^#/, '') || '/';
    const qIdx = path.indexOf('?');
    return {
      pathname: qIdx >= 0 ? path.slice(0, qIdx) : path,
      search: qIdx >= 0 ? path.slice(qIdx) : '',
    };
  }

  function notify() {
    const location = getLocation();
    listeners.forEach((cb) => cb({ location }));
  }

  window.addEventListener('hashchange', notify);

  // Ensure hash has a default route
  if (!window.location.hash) {
    window.location.hash = '#/';
  }

  return {
    get location() { return getLocation(); },
    listen(callback) {
      listeners.push(callback);
      return () => {
        const idx = listeners.indexOf(callback);
        if (idx >= 0) listeners.splice(idx, 1);
      };
    },
    push(path) {
      window.location.hash = '#' + path;
    },
    replace(path) {
      const url = window.location.pathname + window.location.search + '#' + path;
      window.history.replaceState(null, '', url);
      // Dispatch hashchange so Sidebar (and any other hash listeners) update;
      // replaceState alone does not fire this event
      window.dispatchEvent(new HashChangeEvent('hashchange'));
    },
  };
}

const hashHistory = createHashHistory();

/** Catches render errors so a single broken page doesn't blank the whole app. */
class ErrorBoundary extends Component {
  state = { error: null };
  componentDidCatch(error) { this.setState({ error }); }
  render() {
    if (this.state.error) {
      return (
        <ErrorState
          error={this.state.error}
          onRetry={() => this.setState({ error: null })}
        />
      );
    }
    return this.props.children;
  }
}

export default function App() {
  useEffect(() => {
    connectWebSocket();
    return () => disconnectWebSocket();
  }, []);

  return (
    <div class="min-h-screen bg-gray-50">
      <Sidebar />

      {/* Content area: offset for sidebar on desktop, bottom padding for tab bar on mobile */}
      <main class="md:ml-60 pb-16 md:pb-0 min-h-screen">
        <div class="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <ErrorBoundary>
            <Router history={hashHistory}>
              <Home path="/" />
              <Discovery path="/discovery" />
              <Capabilities path="/capabilities" />
              <Predictions path="/predictions" />
              <Patterns path="/patterns" />
              <Automations path="/automations" />
              <Intelligence path="/intelligence" />
              <Shadow path="/shadow" />
              <Settings path="/settings" />
              <DataCuration path="/data-curation" />
            </Router>
          </ErrorBoundary>
        </div>
      </main>
    </div>
  );
}
