import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Dashboard from './pages/Dashboard';
import MemeAnalyzer from './pages/MemeAnalyzer';
import ModelArchitecture from './pages/ModelArchitecture';
import ModelComparison from './pages/ModelComparison';
import AboutProject from './pages/AboutProject';

export default function App() {
  return (
    <div className="min-h-screen bg-[#F7F8F5] flex flex-col font-sans text-[#172923]">
      {/* Mindful-Inspired Sticky Header */}
      <Navbar />

      {/* Main Content Area */}
      <main className="flex-1 w-full">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyzer" element={<MemeAnalyzer />} />
          <Route path="/architecture" element={<ModelArchitecture />} />
          <Route path="/comparison" element={<ModelComparison />} />
          <Route path="/about" element={<AboutProject />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      {/* Mindful-Inspired Dark Forest Footer */}
      <Footer />
    </div>
  );
}
