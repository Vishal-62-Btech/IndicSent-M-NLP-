import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import { BackendProvider } from './context/BackendContext.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <BackendProvider>
        <App />
      </BackendProvider>
    </BrowserRouter>
  </React.StrictMode>,
)
