import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { checkBackendHealth, fetchSupportedModels, API_BASE_URL } from '../services/api';

const BackendContext = createContext(null);

export function BackendProvider({ children }) {
  const [status, setStatus] = useState('checking'); // 'checking' | 'online' | 'offline'
  const [healthData, setHealthData] = useState(null);
  const [modelsCatalog, setModelsCatalog] = useState(null);
  const [lastChecked, setLastChecked] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const refreshStatus = useCallback(async () => {
    setStatus('checking');
    setErrorMessage(null);

    try {
      const [health, models] = await Promise.all([
        checkBackendHealth(),
        fetchSupportedModels().catch(() => null),
      ]);

      setHealthData(health);
      if (models?.modes) {
        setModelsCatalog(models.modes);
      }
      setStatus('online');
      setLastChecked(new Date());
    } catch (err) {
      console.warn('Backend connection failed:', err.message);
      setStatus('offline');
      setHealthData(null);
      setErrorMessage(err.message);
      setLastChecked(new Date());
    }
  }, []);

  useEffect(() => {
    refreshStatus();

    // Periodic heartbeat check every 30 seconds
    const interval = setInterval(refreshStatus, 30000);
    return () => clearInterval(interval);
  }, [refreshStatus]);

  return (
    <BackendContext.Provider
      value={{
        status,
        isOnline: status === 'online',
        isChecking: status === 'checking',
        healthData,
        modelsCatalog,
        lastChecked,
        errorMessage,
        refreshStatus,
        apiBaseUrl: API_BASE_URL,
      }}
    >
      {children}
    </BackendContext.Provider>
  );
}

export function useBackend() {
  const context = useContext(BackendContext);
  if (!context) {
    throw new Error('useBackend must be used within a BackendProvider');
  }
  return context;
}
