import '@mantine/core/styles.css'
import { MantineProvider } from '@mantine/core'
import { theme } from './theme'
import { DragDrop } from './components/DragDrop'
import { FileList } from './components/FileList'
import { useState } from 'react'
import { Button } from '@mantine/core'
import { IconSettingsCog } from '@tabler/icons-react'
import { QuestionTable } from './components/QuestionTable'
import { v4 as uuidv4 } from 'uuid'

import './App.css'
import { API_URL } from './env.ts'

export interface Question {
  id: string
  text: string
  score: number
}

export default function App() {
  const [files, setFiles] = useState<File[]>([])
  const [selection, setSelection] = useState<string[]>([])
  const [questions, setQuestions] = useState<Question[]>([])
  const [loading, setLoading] = useState<boolean>(false)

  const generateQuestions = async () => {
    setLoading(true)
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })

    try {
      const response = await fetch(`${API_URL}/generate_questions`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        const questionsWithId = data.questions.map(
          (question: { text: string; similarity_score: number }) => {
            return {
              id: uuidv4(),
              text: question.text,
              score: question.similarity_score,
            }
          }
        )
        setQuestions(questionsWithId)
      }
    } catch (error) {
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <MantineProvider theme={theme}>
      <div className="body">
        <div className="upload-container">
          <h2>Upload videos / supplementary documents</h2>

          <div className="video-table-container">
            <DragDrop
              files={files}
              setFiles={setFiles}
            />

            <FileList
              files={files}
              setFiles={setFiles}
            />
          </div>
        </div>

        <div className="generate-container">
          <Button
            loading={loading}
            onClick={generateQuestions}
            disabled={files.length === 0 || loading}
            className="generate-question-btn"
          >
            <IconSettingsCog className="setting-icon" /> Generate Questions
          </Button>
        </div>

        <div className="question-container">
          <h2>Generated Questions</h2>

          <QuestionTable
            selection={selection}
            setSelection={setSelection}
            questions={questions}
          />
        </div>
      </div>
    </MantineProvider>
  )
}
