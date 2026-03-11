/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { FileText, Code, Globe, ExternalLink } from 'lucide-react';

interface Model {
  id: string;
  title: string;
}

interface Case {
  id: string;
  text: string;
  models: Model[];
}

const NAV_LINKS = [
  { name: 'Paper', url: 'https://assasinatee.github.io/AlignAudio/', icon: <FileText className="w-4 h-4" /> },
  { name: 'Code', url: 'https://github.com/assasinatee/AlignAudio', icon: <Code className="w-4 h-4" /> },
  // { name: 'Project Page', url: '#', icon: <Globe className="w-4 h-4" /> },
  // { name: 'arXiv', url: '#', icon: <ExternalLink className="w-4 h-4" /> },
];


const CASES: Case[] = [
  { id: 'Case1', text: 'A woman talking followed by a group of people laughing as plastic crinkles.', 
    models: [{ id: '1', title: 'STAR' }, { id: '2', title: 'ST5 + FM' }, { id: '3', title: 'GT → ST5 + FM' }, { id: '4', title: 'AlignAudio' }] },
  { id: 'Case2', text: 'A speedboat is racing across water with loud wind noise.', 
    models: [{ id: '1', title: 'STAR' }, { id: '2', title: 'ST5 + FM' }, { id: '3', title: 'GT → ST5 + FM' }, { id: '4', title: 'AlignAudio' }] },
  { id: 'Case3', text: 'A woman talking followed by a group of people laughing as plastic crinkles.', 
    models: [{ id: '1', title: 'STAR' }, { id: '2', title: 'ST5 + FM' }, { id: '3', title: 'GT → ST5 + FM' }, { id: '4', title: 'AlignAudio' }] },
  { id: 'Case4', text: 'A woman talks and a baby whispers.', 
    models: [{ id: '1', title: 'STAR' }, { id: '2', title: 'ST5 + FM' }, { id: '3', title: 'GT → ST5 + FM' }, { id: '4', title: 'AlignAudio' }] },
  { id: 'Case5', text: 'A man speaks and a vehicle passes.', 
    models: [{ id: '1', title: 'STAR' }, { id: '2', title: 'ST5 + FM' }, { id: '3', title: 'GT → ST5 + FM' }, { id: '4', title: 'AlignAudio' }] },
  { id: 'Case6', text: 'Wind is blowing and heavy rain is falling and splashing.', 
    models: [{ id: '1', title: 'STAR' }, { id: '2', title: 'ST5 + FM' }, { id: '3', title: 'GT → ST5 + FM' }, { id: '4', title: 'AlignAudio' }] },
  { id: 'Case7', text: 'Speech followed by quietness and a man speaks and laughs.', 
    models: [{ id: '1', title: 'STAR' }, { id: '2', title: 'ST5 + FM' }, { id: '3', title: 'GT → ST5 + FM' }, { id: '4', title: 'AlignAudio' }] },
  { id: 'Case8', text: 'A male voice and a machine buzzing.', 
    models: [{ id: '1', title: 'STAR' }, { id: '2', title: 'ST5 + FM' }, { id: '3', title: 'GT → ST5 + FM' }, { id: '4', title: 'AlignAudio' }] },
  { id: 'Case9', text: 'A long burp ends in a sigh.', 
    models: [{ id: '1', title: 'STAR' }, { id: '2', title: 'ST5 + FM' }, { id: '3', title: 'GT → ST5 + FM' }, { id: '4', title: 'AlignAudio' }] },
  { id: 'Case10', text: 'A gun cocking then firing as metal clanks on a hard surface followed by a man talking during an electronic laser effect as gunshots and explosions go off in the distance.', 
    models: [{ id: '1', title: 'STAR' }, { id: '2', title: 'ST5 + FM' }, { id: '3', title: 'GT → ST5 + FM' }, { id: '4', title: 'AlignAudio' }] }
];

export default function App() {
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header Section */}
      <header className="text-center mb-16">
        <h1 className="text-5xl font-bold tracking-tight text-slate-900 mb-4">
          AlignAudio
        </h1>
        <h2 className="text-2xl font-medium text-slate-600 mb-8 max-w-4xl mx-auto leading-relaxed">
          AlignAudio: Dual-Alignment for Noise-Robust Speech-to-Audio Generation
        </h2>
        
        {/* Navigation Links */}
        <nav className="flex items-center justify-center gap-4 text-sm font-medium">
          {NAV_LINKS.map((link, idx) => (
            <React.Fragment key={link.name}>
              <a 
                href={link.url}
                className="flex items-center gap-1.5 text-blue-600 hover:text-blue-800 transition-colors px-2 py-1 rounded-md hover:bg-blue-50"
              >
                {link.icon}
                {link.name}
              </a>
              {idx < NAV_LINKS.length - 1 && (
                <span className="text-slate-300">|</span>
              )}
            </React.Fragment>
          ))}
        </nav>
      </header>

      {/* Abstract Section */}
      <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200 mb-12">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <span className="w-1 h-6 bg-blue-600 rounded-full"></span>
          Abstract
        </h3>
        <p className="text-slate-700 leading-relaxed text-lg">
          <strong>Speech-to-audio (STA)</strong> generation directly maps speech to environmental audio, offering a low-latency alternative to cascaded ASR-TTA systems. Recent end-to-end STA frameworks have demonstrated generation fidelity under clean conditions, but they do not address distribution mismatch from environmental noise, which may degrade fidelity and omit acoustic events. To bridge this gap, we propose <strong>AlignAudio</strong>, a dual-alignment framework for noise-robust end-to-end STA. It (i) aligns representations of clean and noisy speech to preserve semantic cues, and (ii) enforces generation-level consistency in the flow-matching process to maintain temporal coherence. Experiments demonstrate that AlignAudio maintains performance comparable to baselines on clean speech, while outperforming them on all noisy conditions, achieving better temporal coherence and fewer distorted acoustic events.
        </p>
      </section>

      {/* Samples Section */}
      <div className="space-y-12">
        {CASES.map((item) => (
          <div key={item.id} className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="p-6 border-bottom border-slate-100 bg-slate-50/50">
              <div className="flex items-center gap-3 mb-2">
                <span className="px-2.5 py-0.5 rounded-full bg-blue-100 text-blue-700 text-xs font-bold uppercase tracking-wider">
                  {item.id}
                </span>
              </div>
              <p className="text-lg text-slate-700 font-medium">
                "{item.text}"
              </p>
            </div>

            <div className="p-6 space-y-8">
              {['clean', 'noisy'].map((condition) => (
                <div key={condition} className="space-y-4">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${condition === 'clean' ? 'bg-emerald-500' : 'bg-amber-500'}`}></div>
                    <h4 className="text-sm font-bold uppercase tracking-widest text-slate-500">
                      {condition === 'clean' ? 'Clean Condition' : 'Noisy Condition'}
                    </h4>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                    {/* Input Speech */}
                    <div className="bg-slate-50 rounded-xl p-4 border border-dashed border-slate-300">
                      <span className="block text-xs font-bold text-slate-400 uppercase mb-2">Input Speech</span>
                      <audio controls className="w-full h-8" src={`samples/${item.id}/${condition}_speech.wav`} />
                    </div>

                    {/* Models */}
                    {item.models.map((model) => (
                      <div 
                        key={model.id} 
                        className={`rounded-xl p-4 border transition-all ${
                          model.id === '4' 
                            ? 'bg-blue-50 border-blue-200 ring-1 ring-blue-200' 
                            : 'bg-white border-slate-100'
                        }`}
                      >
                        <span className={`block text-xs font-bold uppercase mb-2 ${
                          model.id === '4' ? 'text-blue-600' : 'text-slate-400'
                        }`}>
                          {model.title}
                          {model.id === '4' && <span className="ml-1 text-[10px] bg-blue-600 text-white px-1 rounded">Ours</span>}
                        </span>
                        
                        {condition === 'clean' && model.id === '3' ? (
                          <div className="h-8 flex items-center justify-center text-[10px] text-slate-400 text-center leading-tight">
                            Not processed by SE module
                          </div>
                        ) : (
                          <audio controls className="w-full h-8" src={`samples/${item.id}/${condition}_${model.id}.wav`} />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <footer className="mt-20 pb-12 text-center text-slate-400 text-sm">
        <p>© 2026 AlignAudio Project. All rights reserved.</p>
      </footer>
    </div>
  );
}
